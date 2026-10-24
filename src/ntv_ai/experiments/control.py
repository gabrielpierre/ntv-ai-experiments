"""Framework-independent stopping and LR policy, driven only by source validation."""
import math
from copy import deepcopy


DEFAULT_CONTROL = dict(warmup_epochs=3, warmup_start_factor=0.1, patience=25, min_delta=0.001,
                       plateau_patience=10, factor=0.5, min_factor=0.01, keep_resume=2)


def validate_control(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULT_CONTROL):
        raise ValueError("Controle comum requer todos os parâmetros explícitos")
    for key in ('warmup_epochs', 'patience', 'plateau_patience', 'keep_resume'):
        if type(value[key]) is not int or value[key] < (0 if key == 'warmup_epochs' else 1):
            raise ValueError(f'Controle inválido: {key}')
    for key in ('warmup_start_factor', 'factor', 'min_factor', 'min_delta'):
        v = value[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1:
            raise ValueError(f'Controle inválido: {key}')
    if value['factor'] == 1 or value['keep_resume'] < 2:
        raise ValueError('Redução de LR e retenção inválidas')


class TrainingControl:
    def __init__(self, config, state=None):
        validate_control(config)
        self.config = deepcopy(config)
        self.state = deepcopy(state) if state else dict(epoch=-1, reference=None, bad_epochs=0,
            plateau_epochs=0, lr_factor=1.0, steps=0, base_lrs=[], warmup_steps=None, stopped=False)
        self.optimizer = None

    def bind(self, optimizer, steps_per_epoch):
        self.optimizer = optimizer
        if not self.state['base_lrs']:
            self.state['base_lrs'] = [float(g['lr']) for g in optimizer.param_groups]
            self.state['warmup_steps'] = self.config['warmup_epochs'] * steps_per_epoch
        if len(self.state['base_lrs']) != len(optimizer.param_groups):
            raise ValueError('Grupos do otimizador incompatíveis com a retomada')
        optimizer.register_step_pre_hook(self.before_step)
        optimizer.register_step_post_hook(self.after_step)
        self.apply_rates()

    def apply_rates(self):
        if self.optimizer is None:
            return
        warmup = self.state['warmup_steps'] or 0
        progress = min(self.state['steps'] / warmup, 1) if warmup else 1
        warm = self.config['warmup_start_factor'] + (1-self.config['warmup_start_factor']) * progress
        for group, base in zip(self.optimizer.param_groups, self.state['base_lrs'], strict=True):
            group['lr'] = base * warm * self.state['lr_factor']

    def before_step(self, optimizer, args, kwargs):
        self.apply_rates()

    def after_step(self, optimizer, args, kwargs):
        if float(getattr(optimizer, 'found_inf', 0)) == 0:
            self.state['steps'] += 1

    def observe(self, epoch, metric):
        if not isinstance(metric, (float, int)) or not math.isfinite(metric) or not 0 <= metric <= 1:
            raise ValueError('mAP50_95 comum inválido')
        if epoch <= self.state['epoch']:
            raise ValueError('Controle requer épocas crescentes')
        self.state['epoch'] = epoch
        reference = self.state['reference']
        warmup = epoch < self.config['warmup_epochs']
        improved = reference is None or metric > reference + self.config['min_delta']
        if improved:
            self.state['reference'] = metric
            self.state['bad_epochs'] = self.state['plateau_epochs'] = 0
        else:
            self.state['bad_epochs'] += 1
            if not warmup:
                self.state['plateau_epochs'] += 1
            if self.state['plateau_epochs'] >= self.config['plateau_patience']:
                self.state['lr_factor'] = max(self.config['min_factor'], self.state['lr_factor'] * self.config['factor'])
                self.state['plateau_epochs'] = 0
        self.state['stopped'] = self.state['bad_epochs'] >= self.config['patience']
        self.apply_rates()
        return self.state['stopped']


class CommonEarlyStop(Exception):
    """Internal normal termination for a native loop without a stop flag."""


def retain_checkpoints(latest, keep):
    """Retain complete native states and their matching controller sidecars."""
    checkpoints = sorted(latest.parent.glob('resume-*' + latest.suffix))
    for old in checkpoints[:-keep]:
        old.unlink()
        old.with_suffix(old.suffix + '.control.json').unlink(missing_ok=True)
