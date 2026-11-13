"""Exercise actual Git commits and subprocesses without an API or ML dependencies."""
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from crude_production_autoresearch.research import EDITABLE, ROOT, RequestLimiter, apply_proposal, git, main, propose, retry_delay, run, train


def test_heartbeat_stops_and_log_closes_even_on_error(tmp_path):
    import threading
    from crude_production_autoresearch.research import activity, progress, progress_log, LOGGER
    heartbeat_seen = threading.Event()
    before_threads = set(threading.enumerate())
    def capture(message):
        if 'sigue en curso' in message:
            heartbeat_seen.set()
    with patch('crude_production_autoresearch.research.progress', side_effect=capture):
        with pytest.raises(RuntimeError, match='test'):
            with activity('API de prueba', interval=.01):
                assert heartbeat_seen.wait(2)
                heartbeat_threads = set(threading.enumerate()) - before_threads
                raise RuntimeError('test')
    assert heartbeat_threads
    assert all(not thread.is_alive() for thread in heartbeat_threads)
    output = tmp_path / 'trial'
    initial_handlers = LOGGER.handlers.copy()
    with pytest.raises(RuntimeError, match='test'):
        with progress_log(output):
            progress('Entrenamiento iniciado')
            raise RuntimeError('test')
    assert 'Entrenamiento iniciado' in (tmp_path / 'trial.log').read_text(encoding='utf-8')
    assert LOGGER.handlers == initial_handlers


@pytest.fixture
def research(tmp_path):
    research = SimpleNamespace(root=tmp_path / 'repo', output=tmp_path / 'trial')
    research.root.mkdir()
    git(research.root, 'init')
    git(research.root, 'config', 'user.name', 'Research Test')
    git(research.root, 'config', 'user.email', 'research@example.invalid')
    git(research.root, 'config', 'core.autocrlf', 'false')
    package = research.root / 'src/crude_production_autoresearch'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'features.py').write_text('def build_features(frame):\n    return frame\n')
    (package / 'modeling.py').write_text('def build_model(columns, config):\n    return 0.5\n')
    (package / 'history.py').write_text('# fixed history\n')
    # A deliberately small training entrypoint lets us test the real subprocess/Git path.
    (package / 'train.py').write_text('''import argparse, csv, json
from pathlib import Path
from .modeling import build_model
parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
data = Path(config['data_dir'])
assert not (data / 'test_X.parquet').exists()
out = Path(config['output_root']) / 'result'
out.mkdir(parents=True)
with (out / 'results.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['model', 'mae', 'rmse'])
    writer.writerow(['constant', 0.8, 0.1])
    writer.writerow(['candidate', build_model([], config), 0.5])
(out.parent / 'latest.json').write_text(json.dumps({'run_dir': str(out)}))
''')
    (research.root / 'configs').mkdir()
    (research.root / 'configs/baselines.json').write_text(json.dumps({'data_dir': 'data/processed'}))
    (research.root / 'program.md').write_text((ROOT / 'program.md').read_text(encoding='utf-8'), encoding='utf-8')
    (research.root / '.gitignore').write_text('data/processed/\n__pycache__/\n')
    data = research.root / 'data/processed'
    data.mkdir(parents=True)
    for split in ('train', 'validation', 'test'):
        for suffix in ('X', 'y', 'metadata'):
            (data / f'{split}_{suffix}.parquet').write_bytes(b'frozen input')
    (data / 'manifest.json').write_text('{}')
    git(research.root, 'add', '.')
    git(research.root, 'commit', '-m', 'Initial baselines')
    research.initial = git(research.root, 'rev-parse', 'HEAD')
    return research


def make_proposal(value):
    return {'hypothesis': f'Try {value}',
            'files': {EDITABLE[1]: f'def build_model(columns, config):\n    return {value}\n'}}


def test_regression_error_above_one_is_valid_and_discarded(research):
    run(research.root, research.output, lambda context: make_proposal(123.0), experiments=1)
    entries = [json.loads(line) for line in (research.output / 'history.jsonl').read_text().splitlines()]
    assert entries[1]['status'] == 'discard'
    assert entries[1]['metrics']['candidate']['mae'] == 123.0


def test_old_classification_protocol_cannot_resume(research):
    run(research.root, research.output, lambda context: None, experiments=0)
    path = research.output / 'protocol.json'
    protocol = json.loads(path.read_text())
    protocol['primary_metric'] = 'validation log_loss'
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='Different task/metric'):
        run(research.root, research.output, lambda context: None, experiments=0, resume=True)


def test_real_commits_keep_discard_tie_failure_and_next_parent(research):
    proposals = [make_proposal(.4), make_proposal(.6), make_proposal(.4),
                 make_proposal("float('nan')"), make_proposal(.3)]
    proposals[2]['files'][EDITABLE[1]] += '# A different candidate with the same score\n'
    contexts = []

    def proposer(context):
        contexts.append(context)
        return proposals.pop(0)

    summary = run(research.root, research.output, proposer, experiments=5, timeout=30)
    history = [json.loads(line) for line in (research.output / 'history.jsonl').read_text().splitlines()]
    assert [e['status'] for e in history] == ['baseline', 'keep', 'discard', 'discard', 'failed', 'keep']
    assert history[0]['metrics']['constant']['mae'] == .8
    assert history[2]['parent_commit'] == history[1]['commit']
    assert history[5]['parent_commit'] == history[1]['commit']
    assert summary['best_mae'] == .3
    assert git(Path(summary['worktree']), 'rev-parse', 'HEAD') == summary['best_commit']
    assert git(research.root, 'rev-parse', 'HEAD') == research.initial
    assert git(research.root, 'status', '--porcelain') == ''
    assert len(git(Path(summary['worktree']), 'log', '--oneline').splitlines()) == 3
    assert 'return 0.4' in contexts[2]['files'][EDITABLE[1]]
    assert 'baselines' in contexts[0]['instructions']
    assert 'test' not in contexts[0]['data']['splits']
    assert git(research.root, 'show', history[2]['commit'] + ':' + EDITABLE[1]).splitlines()[-1] == '    return 0.6'
    log = research.output.with_suffix('.log').read_text(encoding='utf-8')
    for message in ('Ensayo 0:', 'Entrenamiento: iniciando subproceso',
                    'validando propuesta', 'candidato MAE=0.300000',
                    '5: keep', 'Investigación finalizada: completed'):
        assert message in log


def read_entries(output):
    return [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]


def test_resume_keeps_best_history_and_adds_attempts_with_new_provider(research):
    first = run(research.root, research.output, lambda context: make_proposal(.4), experiments=1,
                provider='gemini', model='first-model')
    baseline = (research.output / 'run_000/training.log').stat().st_mtime_ns
    previous_log = research.output.with_suffix('.log').read_text(encoding='utf-8')
    contexts = []
    proposals = iter([make_proposal(.6), make_proposal(.3)])
    def proposer(context):
        contexts.append({**context, 'history': list(context['history'])})
        return next(proposals)
    summary = run(research.root, research.output, proposer, experiments=2, resume=True,
                  provider='groq', model='second-model')
    entries = read_entries(research.output)
    assert [e['experiment'] for e in entries] == [0, 1, 2, 3]
    assert [e['status'] for e in entries] == ['baseline', 'keep', 'discard', 'keep']
    assert contexts[0]['best_score'] == .4
    assert len(contexts[0]['history']) == 2
    assert 'return 0.4' in contexts[0]['files'][EDITABLE[1]]
    assert entries[2]['parent_commit'] == first['best_commit']
    assert entries[3]['parent_commit'] == first['best_commit']
    assert entries[1]['provider'] == 'gemini'
    assert entries[3]['provider'] == 'groq'
    assert summary['best_mae'] == .3
    assert baseline == (research.output / 'run_000/training.log').stat().st_mtime_ns
    assert len(list((research.output / 'executions').glob('*.json'))) == 2
    assert git(research.root, 'status', '--porcelain') == ''
    log = research.output.with_suffix('.log').read_text(encoding='utf-8')
    assert log.startswith(previous_log)
    assert 'Reanudando investigación' in log


def test_resume_recovers_history_when_summary_write_was_interrupted(research):
    from crude_production_autoresearch.research import write_json
    def write(path, value):
        if path.name == 'summary.json' and value['best_mae'] == .4:
            raise OSError('Simulated checkpoint interruption')
        write_json(path, value)
    with patch('crude_production_autoresearch.research.write_json', side_effect=write):
        with pytest.raises(OSError, match='checkpoint interruption'):
            run(research.root, research.output, lambda context: make_proposal(.4), experiments=1)
    summary = run(research.root, research.output, lambda context: make_proposal(.3),
                  experiments=1, resume=True)
    assert [e['status'] for e in read_entries(research.output)] == ['baseline', 'keep', 'keep']
    assert summary['best_mae'] == .3


def test_resume_records_pending_attempt_and_preserves_its_artifacts(research):
    from crude_production_autoresearch.research import write_json
    summary = run(research.root, research.output, lambda context: None, experiments=0)
    worktree = Path(summary['worktree'])
    apply_proposal(worktree, make_proposal(.2))
    git(worktree, 'add', *EDITABLE)
    git(worktree, 'commit', '-m', 'Unconfirmed candidate')
    write_json(research.output / 'pending.json', {'experiment': 1, 'hypothesis': 'Unconfirmed',
               'parent_commit': summary['best_commit'], 'commit': git(worktree, 'rev-parse', 'HEAD'),
               'metrics': None, 'status': 'running'})
    (research.output / 'run_001').mkdir()
    artifact = research.output / 'run_001/training.log'
    artifact.write_text('Partial training')
    result = run(research.root, research.output, lambda context: make_proposal(.4), experiments=1, resume=True)
    assert [e['status'] for e in read_entries(research.output)] == ['baseline', 'interrupted', 'keep']
    assert result['attempts'] == 2
    assert artifact.read_text() == 'Partial training'
    assert read_entries(research.output)[2]['parent_commit'] == research.initial


def test_resume_rejects_changed_frozen_data_and_protocol(research):
    run(research.root, research.output, lambda context: None, experiments=0)
    path = research.output / 'data/train_X.parquet'
    previous = path.read_bytes()
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='frozen data changed'):
        run(research.root, research.output, lambda context: None, resume=True)
    path.write_bytes(previous)
    (research.root / 'program.md').write_text('Different instructions')
    git(research.root, 'add', 'program.md')
    git(research.root, 'commit', '-m', 'Change protocol')
    with pytest.raises(ValueError, match='Protocol differs'):
        run(research.root, research.output, lambda context: None, resume=True)
    assert len(read_entries(research.output)) == 1


def test_research_cannot_run_twice_at_the_same_time(research):
    from crude_production_autoresearch.research import research_lock
    with research_lock(research.output):
        with pytest.raises(ValueError, match='already running'):
            run(research.root, research.output, lambda context: None, experiments=0)
    assert not (research.output.exists())
    run(research.root, research.output, lambda context: None, experiments=0)


def test_interrupt_stops_and_can_be_resumed(research):
    def interrupt(context):
        raise KeyboardInterrupt()
    result = run(research.root, research.output, interrupt, experiments=3)
    assert result['attempts'] == 1
    assert read_entries(research.output)[1]['status'] == 'interrupted'
    summary = run(research.root, research.output, lambda context: make_proposal(.4), experiments=1, resume=True)
    assert summary['attempts'] == 2
    assert summary['best_mae'] == .4


def test_daily_quota_stops_without_spending_remaining_attempts(research):
    from io import BytesIO
    from urllib.error import HTTPError
    error = HTTPError('https://api.groq.com', 429, 'Too Many Requests', {},
                      BytesIO(b'{"error":{"message":"Rate limit reached for requests per day (RPD)"}}'))
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
            patch('crude_production_autoresearch.research.urlopen', side_effect=error) as request, \
            patch('crude_production_autoresearch.research.time.sleep') as sleep:
        summary = run(research.root, research.output, lambda context: propose(context), experiments=5)
    assert summary['attempts'] == 1
    assert read_entries(research.output)[1]['status'] == 'quota_exhausted'
    request.assert_called_once()
    # subprocess.wait also uses time.sleep for polling; no API backoff occurred.
    assert all(call.args[0] < 2 for call in sleep.call_args_list)
    result = run(research.root, research.output, lambda context: make_proposal(.4), experiments=1, resume=True)
    assert result['best_mae'] == .4


@pytest.mark.parametrize('proposal', [
    {'hypothesis': 'escape', 'files': {'../escape.py': 'x=1'}},
    {'hypothesis': 'fixed', 'files': {'src/crude_production_autoresearch/history.py': 'x=1'}},
    {'hypothesis': 'syntax', 'files': {EDITABLE[0]: 'def build_features('}},
    {'hypothesis': 'interface', 'files': {EDITABLE[0]: 'x=1'}},
    {'hypothesis': 'empty', 'files': {}},
], ids=['path-escape', 'protected-file', 'invalid-syntax', 'missing-interface', 'empty-files'])
def test_reject_paths_syntax_missing_interface_and_empty_changes(research, proposal):
    before = (research.root / EDITABLE[0]).read_text()
    with pytest.raises((ValueError, SyntaxError)):
        apply_proposal(research.root, proposal)
    assert (research.root / EDITABLE[0]).read_text() == before


def test_dirty_checkout_rejected_before_creating_worktree(research):
    (research.root / EDITABLE[0]).write_text('# user change')
    with pytest.raises(ValueError, match='Commit or stash'):
        run(research.root, research.output, lambda context: make_proposal(.4))
    assert not (research.output.exists())


def test_missing_data_rejected_before_creating_branch(research):
    (research.root / 'data/processed/validation_X.parquet').unlink()
    with pytest.raises(FileNotFoundError, match='Run preprocessing first'):
        run(research.root, research.output, lambda context: make_proposal(.4))
    assert not (research.output.exists())
    assert 'autoresearch/' not in git(research.root, 'branch', '--list')


def test_invalid_proposal_keeps_hypothesis_without_a_commit(research):
    proposal = {'hypothesis': 'Change protected history',
                'files': {'src/crude_production_autoresearch/history.py': '# change'}}
    summary = run(research.root, research.output, lambda context: proposal, experiments=1, timeout=30)
    entry = json.loads((research.output / 'history.jsonl').read_text().splitlines()[1])
    assert entry['hypothesis'] == proposal['hypothesis']
    assert entry['status'] == 'failed'
    assert entry['commit'] is None
    assert summary['best_commit'] == research.initial


def test_protected_training_write_stops_and_restores_history(research):
    def tamper(worktree, config, directory, timeout):
        (worktree / 'src/crude_production_autoresearch/history.py').write_text('# altered')
        return {'constant': {'mae': .8}, 'candidate': {'mae': .1}}

    with patch('crude_production_autoresearch.research.train', side_effect=tamper):
        summary = run(research.root, research.output, lambda context: make_proposal(.4), experiments=3)
    entries = (research.output / 'history.jsonl').read_text().splitlines()
    assert len(entries) == 1
    assert json.loads(entries[0])['status'] == 'failed'
    assert summary['best_mae'] is None
    assert (Path(summary['worktree']) / 'src/crude_production_autoresearch/history.py').read_text() == '# fixed history\n'


def test_api_failure_is_recorded_and_loop_continues_from_best(research):
    from urllib.error import HTTPError
    calls = []
    def unavailable(context):
        calls.append(context)
        if len(calls) == 1:
            raise HTTPError('https://api.groq.com', 429, 'Quota exceeded', {}, None)
        return make_proposal(.4)

    summary = run(research.root, research.output, unavailable, experiments=2, timeout=30)
    entries = [json.loads(line) for line in (research.output / 'history.jsonl').read_text().splitlines()]
    assert [entry['status'] for entry in entries] == ['baseline', 'failed', 'keep']
    assert entries[2]['parent_commit'] == research.initial
    assert summary['best_mae'] == .4
    assert '429' in entries[1]['error']


def test_api_retry_after_and_backoff_then_success():
    from io import BytesIO
    from urllib.error import HTTPError, URLError
    error = HTTPError('https://api.groq.com', 429, 'Rate limit', {'Retry-After': '7'},
                      BytesIO(b'{"error":{"message":"Wait"}}'))
    response = {'choices': [{'message': {'content': json.dumps(make_proposal(.4))}}]}
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
            patch('crude_production_autoresearch.research.urlopen',
                  side_effect=[error, URLError('Connection lost'), BytesIO(json.dumps(response).encode())]) as request, \
            patch('crude_production_autoresearch.research.time.sleep') as sleep:
        result = propose({'instructions': 'Start with baselines'}, 'test-model')
    assert result == make_proposal(.4)
    assert request.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [7, 4]


def test_api_retry_budget_is_bounded():
    from io import BytesIO
    from urllib.error import HTTPError
    errors = [HTTPError('https://api.groq.com', 503, 'Unavailable', {},
                        BytesIO(b'{"error":{"message":"Temporary failure"}}')) for _ in range(3)]
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
            patch('crude_production_autoresearch.research.urlopen', side_effect=errors) as request, \
            patch('crude_production_autoresearch.research.time.sleep') as sleep:
        with pytest.raises(HTTPError):
            propose({'instructions': 'Start with baselines'}, 'test-model')
    assert request.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [2, 4]


def test_retry_after_http_date_and_invalid_header():
    with patch('crude_production_autoresearch.research.time.time', return_value=0):
        assert retry_delay('Thu, 01 Jan 1970 00:00:09 GMT', 0) == 9
    assert retry_delay('invalid', 1) == 4
    assert retry_delay('nan', 0) == 2


def test_generated_json_failure_can_be_retried():
    from io import BytesIO
    from urllib.error import HTTPError
    error = HTTPError('https://api.groq.com', 400, 'Bad Request', {},
                      BytesIO(b'{"error":{"code":"json_validate_failed","message":"Invalid generated JSON"}}'))
    response = {'choices': [{'message': {'content': json.dumps(make_proposal(.4))}}]}
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
            patch('crude_production_autoresearch.research.urlopen',
                  side_effect=[error, BytesIO(json.dumps(response).encode())]) as request, \
            patch('crude_production_autoresearch.research.time.sleep') as sleep:
        assert propose({'instructions': 'Start with baselines'}, 'test-model') == make_proposal(.4)
    assert request.call_count == 2
    sleep.assert_called_once_with(2)


def test_llm_request_uses_prompt_and_returns_only_proposal():
    from io import BytesIO
    proposal = make_proposal(.4)
    response = {'choices': [{'message': {'content': json.dumps(proposal)}}]}
    context = {'instructions': 'Start with baselines', 'files': {}, 'history': []}
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
            patch('crude_production_autoresearch.research.urlopen',
                  return_value=BytesIO(json.dumps(response).encode())) as request:
        assert propose(context, 'test-model') == proposal
    payload = json.loads(request.call_args.args[0].data)
    assert payload['messages'][0]['content'] == 'Start with baselines'
    assert 'test-key' not in json.dumps(payload)
    assert context['instructions'] == 'Start with baselines'


def test_llm_http_error_shows_provider_detail_without_api_key():
    from io import BytesIO
    from urllib.error import HTTPError
    response = BytesIO(json.dumps({'error': {'message': 'Invalid request for test-key'}}).encode())
    error = HTTPError('https://api.groq.com', 400, 'Bad Request', {}, response)
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}), \
            patch('crude_production_autoresearch.research.urlopen', side_effect=error) as request, \
            patch('crude_production_autoresearch.research.time.sleep') as sleep:
        with pytest.raises(HTTPError) as raised:
            propose({'instructions': 'Start with baselines'}, 'test-model')
    assert '400' in str(raised.value)
    assert 'Invalid request' in str(raised.value)
    assert 'test-key' not in str(raised.value)
    assert request.call_count == 1
    sleep.assert_not_called()


def test_timeout_kills_actual_subprocess(research):
    path = research.root / 'src/crude_production_autoresearch/train.py'
    path.write_text('import time\ntime.sleep(30)\n')
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        train(research.root, {'data_dir': str(research.root / 'data/processed')}, research.output, timeout=1)
    assert time.monotonic() - started < 10


def test_spaces_requests_and_counts_elapsed_training_time():
    now = [0.0]
    def advance(seconds):
        now[0] += seconds
    limiter = RequestLimiter(10)
    with patch('crude_production_autoresearch.research.time.monotonic', side_effect=lambda: now[0]), \
            patch('crude_production_autoresearch.research.time.sleep', side_effect=advance) as sleep:
        limiter.wait()
        sleep.assert_not_called()
        now[0] += 2  # Response/training took two seconds.
        limiter.wait()
        sleep.assert_called_once_with(4)
        assert now[0] == 6
        now[0] += 20  # No additional waiting after a slow experiment.
        limiter.wait()
        assert sleep.call_count == 1
        limiter.wait()
        assert now[0] == 32


@pytest.mark.parametrize('retry_after, expected', [('2', [0, 6, 12]), ('9', [0, 9, 15])])
def test_retries_and_next_proposal_share_limit_and_retry_after(retry_after, expected):
    from io import BytesIO
    from urllib.error import HTTPError
    now, starts = [0.0], []
    proposal = {'hypothesis': 'Test', 'files': {EDITABLE[0]: 'source'}}
    def request(*args, **kwargs):
        starts.append(now[0])
        if len(starts) == 1:
            raise HTTPError('https://api.groq.com', 429, 'Rate limit', {'Retry-After': retry_after},
                            BytesIO(b'{"error":{"message":"Wait"}}'))
        return BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(proposal)}}]}).encode())
    def advance(seconds):
        now[0] += seconds
    limiter = RequestLimiter(10)
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True), \
            patch('crude_production_autoresearch.research.time.monotonic', side_effect=lambda: now[0]), \
            patch('crude_production_autoresearch.research.time.sleep', side_effect=advance), \
            patch('crude_production_autoresearch.research.urlopen', side_effect=request):
        assert propose({'instructions': 'Test'}, limiter=limiter) == proposal
        assert propose({'instructions': 'Test'}, limiter=limiter) == proposal
    assert starts == expected


def test_cli_shares_limiter_between_proposals():
    def fake_run(root, output, proposer, experiments, timeout, **kwargs):
        proposer({'instructions': 'First'})
        proposer({'instructions': 'Second'})
        return {'best_mae': .5}
    with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}, clear=True), \
            patch('sys.argv', ['autoresearch', '--provider', 'gemini', '--rpm', '10']), \
            patch('crude_production_autoresearch.research.run', side_effect=fake_run), \
            patch('crude_production_autoresearch.research.propose') as proposal, \
            patch('builtins.print'):
        main()
    first, second = proposal.call_args_list
    assert first.kwargs['limiter'] is second.kwargs['limiter']
    assert first.kwargs['limiter'].interval == 6


@pytest.mark.parametrize('rpm', ['0', '-1', '1.5', 'nan'])
def test_cli_rejects_invalid_rpm_before_run(rpm):
    with patch('sys.argv', ['autoresearch', '--rpm', rpm]), \
            patch('sys.stderr'), patch('crude_production_autoresearch.research.run') as run_loop:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
        run_loop.assert_not_called()


def test_gemini_request_and_multipart_response():
    from io import BytesIO
    proposal = {'hypothesis': 'New feature', 'files': {EDITABLE[0]: 'def build_features(frame):\n    return frame\n'}}
    content = json.dumps(proposal)
    response = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'thought': True, 'text': 'Internal reasoning'},
        {'text': content[:20]}, {'text': content[20:]}]}}]}
    context = {'instructions': 'Use only past sessions', 'files': {}, 'history': []}
    with patch.dict(os.environ, {'GEMINI_API_KEY': 'gemini-secret'}, clear=True), \
            patch('crude_production_autoresearch.research.urlopen',
                  return_value=BytesIO(json.dumps(response).encode())) as request:
        assert propose(context, provider='gemini') == proposal
    sent = request.call_args.args[0]
    assert sent.full_url == 'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent'
    assert sent.get_header('X-goog-api-key') == 'gemini-secret'
    payload = json.loads(sent.data)
    assert payload['systemInstruction']['parts'][0]['text'] == context['instructions']
    assert json.loads(payload['contents'][0]['parts'][0]['text']) == {'files': {}, 'history': []}
    # TextResponseFormat.MimeType is a protobuf enum, not a MIME string.
    assert payload['generationConfig']['responseFormat']['text']['mimeType'] == 'APPLICATION_JSON'
    assert 'gemini-secret' not in sent.full_url + json.dumps(payload)


def test_gemini_rate_limit_retries_and_redacts_error():
    from io import BytesIO
    from urllib.error import HTTPError
    errors = [HTTPError('https://generativelanguage.googleapis.com', 429, 'Too Many Requests',
                        {'Retry-After': '9'}, BytesIO(b'{"error":{"code":429,"message":"Quota for gemini-secret","status":"RESOURCE_EXHAUSTED"}}'))
              for _ in range(3)]
    with patch.dict(os.environ, {'GEMINI_API_KEY': 'gemini-secret'}, clear=True), \
            patch('crude_production_autoresearch.research.urlopen', side_effect=errors) as request, \
            patch('crude_production_autoresearch.research.time.sleep') as sleep:
        with pytest.raises(HTTPError) as raised:
            propose({'instructions': 'Test'}, provider='gemini')
    assert request.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [9, 9]
    assert 'Quota' in str(raised.value)
    assert 'gemini-secret' not in str(raised.value)


@pytest.mark.parametrize('response, reason', [
    ({'promptFeedback': {'blockReason': 'SAFETY'}}, 'SAFETY'),
    ({'candidates': [{'finishReason': 'MAX_TOKENS'}]}, 'MAX_TOKENS'),
], ids=['blocked', 'truncated'])
def test_gemini_blocked_or_truncated_response_reports_reason(response, reason):
    from io import BytesIO
    with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}, clear=True), \
            patch('crude_production_autoresearch.research.urlopen',
                  return_value=BytesIO(json.dumps(response).encode())):
        with pytest.raises(RuntimeError, match=reason):
            propose({'instructions': 'Test'}, provider='gemini')


def test_provider_requires_its_own_key():
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
        with pytest.raises(ValueError, match='GEMINI_API_KEY'):
            propose({'instructions': 'Test'}, provider='gemini')
    with pytest.raises(ValueError, match='Unknown API provider'):
        propose({}, provider='unknown')


@pytest.mark.parametrize('arguments, env, expected', [
    (['--provider', 'gemini'], {'GEMINI_API_KEY': 'test-key'}, (None, 'gemini')),
    (['--provider', 'gemini', '--model', 'custom-model'], {'GEMINI_API_KEY': 'test-key'}, ('custom-model', 'gemini')),
    ([], {'GROQ_API_KEY': 'test-key'}, (None, 'groq')),
    (['--provider', 'gemini', '--experiments', '0'], {}, None),
], ids=['gemini-default', 'gemini-custom-model', 'groq-default', 'baseline-without-key'])
def test_cli_selects_provider_and_model_and_allows_baseline_without_key(arguments, env, expected):
    def fake_run(root, output, proposer, experiments, timeout, **kwargs):
        if experiments:
            proposer({'instructions': 'Test'})
        return {'best_mae': .5}
    with patch.dict(os.environ, env, clear=True), \
            patch('sys.argv', ['autoresearch', *arguments]), \
            patch('crude_production_autoresearch.research.run', side_effect=fake_run), \
            patch('crude_production_autoresearch.research.propose') as proposal, \
            patch('builtins.print'):
        main()
        if expected:
            proposal.assert_called_once_with({'instructions': 'Test'}, *expected)
        else:
            proposal.assert_not_called()


def test_training_removes_both_api_keys(tmp_path):
    process = Mock()
    process.wait.return_value = 1
    with patch.dict(os.environ, {'GROQ_API_KEY': 'groq-secret', 'GEMINI_API_KEY': 'gemini-secret'}), \
            patch('crude_production_autoresearch.research.subprocess.Popen', return_value=process) as popen:
        with pytest.raises(RuntimeError, match='Training failed'):
            train(tmp_path, {}, tmp_path / 'run', 10)
    assert 'GROQ_API_KEY' not in popen.call_args.kwargs['env']
    assert 'GEMINI_API_KEY' not in popen.call_args.kwargs['env']
