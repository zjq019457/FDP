# PaLu downstream blocker

Official lm-eval zero-shot evaluation is blocked before the first task because `allenai/openbookqa` is not locally cached and the remote Hub connection resets. No scorer, task, sample, model, or schedule was changed. The four ratio-0.5 cells remain pending and must be rerun unchanged after the dataset dependency is supplied.
