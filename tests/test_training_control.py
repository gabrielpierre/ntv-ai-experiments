from copy import deepcopy

import pytest

from ntv_ai.experiments.control import DEFAULT_CONTROL, TrainingControl, validate_control


def test_stops_after_25_and_reductions_do_not_reset_patience():
    c=TrainingControl(DEFAULT_CONTROL)
    for epoch in range(26):
        stopped=c.observe(epoch, .5)
        if epoch==12: assert c.state['lr_factor']==.5
        if epoch==22: assert c.state['lr_factor']==.25
        assert stopped == (epoch==25)
    assert c.state['bad_epochs']==25


def test_threshold_and_resume_preserve_policy():
    a=TrainingControl({**DEFAULT_CONTROL,'warmup_epochs':0})
    a.observe(0,.5)
    for epoch in range(1,13): a.observe(epoch,.5005)
    b=TrainingControl(a.config,a.state)
    for epoch in range(13,40):
        metric=.503 if epoch==16 else .5006
        assert a.observe(epoch,metric)==b.observe(epoch,metric)
        assert a.state==b.state
    with pytest.raises(ValueError,match='crescentes'):a.observe(39,.5)


def test_floor_and_invalid_metrics():
    c=TrainingControl({**DEFAULT_CONTROL,'patience':200,'warmup_epochs':0})
    for i in range(120): c.observe(i,0)
    assert c.state['lr_factor']==.01
    with pytest.raises(ValueError):c.observe(120,float('nan'))
    with pytest.raises(ValueError):validate_control({**DEFAULT_CONTROL,'patience':True})


def test_warmup_rates_preserve_group_ratios_without_torch():
    class Optimizer:
        param_groups=[{'lr':.001},{'lr':.0001}]
        def register_step_pre_hook(self,f):pass
        def register_step_post_hook(self,f):pass
    optimizer=Optimizer();c=TrainingControl(DEFAULT_CONTROL);c.bind(optimizer,2)
    assert optimizer.param_groups[0]['lr']==pytest.approx(.0001)
    for _ in range(6):c.after_step(optimizer,(),{});c.apply_rates()
    assert optimizer.param_groups[0]['lr']==pytest.approx(.001)
    for i in range(13):c.observe(i,.4)
    assert optimizer.param_groups[0]['lr']==pytest.approx(.0005)
    assert optimizer.param_groups[1]['lr']==pytest.approx(.00005)


def test_best_checkpoint_uses_any_gain_and_retention_keeps_two(tmp_path):
    from ntv_ai.experiments.control import retain_checkpoints
    from ntv_ai.experiments.selection import CheckpointSelection
    selection = CheckpointSelection(tmp_path/'selection')
    control = TrainingControl({**DEFAULT_CONTROL, 'warmup_epochs':0})
    for epoch, metric in enumerate([.5,.5005,.5005,.5004,.5003]):
        checkpoint = tmp_path/f'resume-{epoch:04d}.pt'
        checkpoint.write_text(str(epoch))
        checkpoint.with_suffix('.pt.control.json').write_text('{}')
        selection.consider(epoch,metric,checkpoint)
        control.observe(epoch,metric)
        retain_checkpoints(checkpoint,2)
    assert selection.best['epoch'] == 1
    assert control.state['bad_epochs'] == 4
    assert sorted(p.name for p in tmp_path.glob('resume-*.pt')) == ['resume-0003.pt','resume-0004.pt']
    assert len(list(tmp_path.glob('*.control.json'))) == 2
    assert (tmp_path/'selection/best.pt').read_text() == '1'
