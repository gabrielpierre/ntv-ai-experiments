"""Native-training wrapper with durable run references, isolated from driver death."""
import argparse
from pathlib import Path

from ntv_ai.experiments.cross import cross_predict
from ntv_ai.pipelines.experiments import launch
from ntv_ai.storage import atomic_json, load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    request = load_json(parser.parse_args().request)
    output = Path(request['output'])
    previous_path = output / 'attempts.json'
    attempts = load_json(previous_path)['runs'] if previous_path.exists() else []
    # Recover a finished native run before scheduling another one.
    if attempts:
        prior = Path(attempts[-1])
        if (prior / 'result.json').exists() and load_json(prior / 'status.json')['state'] == 'completed':
            atomic_json(output / 'result.json', load_json(prior / 'result.json'), replace=True)
            return
    def created(path):
        attempts.append(str(path))
        atomic_json(previous_path, {'runs': attempts}, replace=True)
    execution = {'cpu_threads': 8, 'mode': 'workshop_serial'}
    if request['kind'] == 'train':
        resume = attempts[-1] if attempts and (Path(attempts[-1])/'progress.json').exists() else None
        result = launch('train', request['config'], runs_root=output/'native', resume=resume,
                        execution=execution, on_created=created)
    else:
        source = load_json(Path(request['source_result']))
        result = cross_predict(source['run_path'], request['target'], runs_root=output/'native',
                               execution=execution, on_created=created)
        if result.get('state', 'completed') != 'completed':
            raise ValueError(f'Blocked target: {result}')
    atomic_json(output / 'result.json', result, replace=True)


if __name__ == '__main__':
    main()
