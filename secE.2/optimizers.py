from torch.optim import Optimizer
import math
import torch
import time

class AdamW(Optimizer):
    """Implements AdamW algorithm.
    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. Fixing Weight Decay Regularization in Adam:
    https://arxiv.org/abs/1711.05101
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr, eps, amsgrad = group['lr'], group['eps'], group['amsgrad']
            for p in group['params']:
                if p.grad is None:
                    continue

                if group['weight_decay'] != 0:
                    p.data.mul_(1. - lr * group['weight_decay'])

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of learning rates
                    state['exp_avg_lr_1'] = 0.
                    state['exp_avg_lr_2'] = 0.
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    state['PopArt_rescale'] = 1.

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    if not 'max_exp_avg_sq' in state: 
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1

                # We incorporate the term group['lr'] into the momentum, and define the bias_correction1 such that it respects the possibly moving group['lr']
                state['exp_avg_lr_1'] = state['exp_avg_lr_1'] * beta1 + (1. - beta1) * lr
                state['exp_avg_lr_2'] = state['exp_avg_lr_2'] * beta2 + (1. - beta2)
                #bias_correction1 = state['exp_avg_lr_1'] / group['lr'] if group['lr']!=0. else 1. #1. - beta1 ** state['step']
                bias_correction2 = state['exp_avg_lr_2']

                # For convenience, we directly use "sqrt_bias_correction2" and "step_size" as the following
                sqrt_bias_correction2 = math.sqrt(bias_correction2)
                # when state['exp_avg_lr_1'] is zero, exp_avg should also be zero and it is trivial
                one_over_bias_correction1 = lr / state['exp_avg_lr_1'] if state['exp_avg_lr_1']!=0. else 0. 
                step_size = one_over_bias_correction1 * sqrt_bias_correction2 # instead of correcting "denom" by dividing it, we put the correction factor into "step_size" and "eps"


                # Decay the first and second moment running average coefficient
                rescaling = state['PopArt_rescale']
                exp_avg.mul_(beta1*rescaling).add_(grad, alpha=(1. - beta1) * lr)
                exp_avg_sq.mul_(beta2*rescaling**2).addcmul_(grad, grad, value=1. - beta2) 
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(eps * sqrt_bias_correction2) 
                    # 'eps' is first multiplied by sqrt_bias_correction2 and then divided by sqrt_bias_correction2
                else:
                    denom = exp_avg_sq.sqrt().add_(eps * sqrt_bias_correction2)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class AdamBelief(Optimizer):
    """Implements AdamW algorithm.
    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. Fixing Weight Decay Regularization in Adam:
    https://arxiv.org/abs/1711.05101
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr, eps, amsgrad = group['lr'], group['eps'], group['amsgrad']
            for p in group['params']:
                if p.grad is None:
                    continue

                if group['weight_decay'] != 0:
                    p.data.mul_(1. - lr * group['weight_decay'])

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of learning rates
                    state['exp_avg_lr_1'] = 0.
                    state['exp_avg_lr_2'] = 0.
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                if amsgrad:
                    if not 'max_exp_avg_var' in state: 
                        state['max_exp_avg_var'] = torch.zeros_like(p.data)
                    max_exp_avg_var = state['max_exp_avg_var']

                state['step'] += 1

                # We define the bias_correction1 such that it respects the possibly moving "group['lr']"
                state['exp_avg_lr_1'] = state['exp_avg_lr_1'] * beta1 + (1. - beta1) * lr
                state['exp_avg_lr_2'] = state['exp_avg_lr_2'] * beta2 + (1. - beta2)
                #bias_correction1 = state['exp_avg_lr_1'] / group['lr'] if group['lr']!=0. else 1. #1. - beta1 ** state['step']
                bias_correction2 = state['exp_avg_lr_2']

                # For convenience, we directly use "sqrt_bias_correction2" and "step_size" as the following
                sqrt_bias_correction2 = math.sqrt(bias_correction2)
                # when state['exp_avg_lr_1'] is zero, exp_avg should also be zero and it is trivial
                one_over_bias_correction1 = lr / state['exp_avg_lr_1'] if state['exp_avg_lr_1']!=0. else 0. 
                step_size = one_over_bias_correction1 * sqrt_bias_correction2 # instead of correcting "denom" by dividing it, we put the correction factor into "step_size" and "eps"


                # Decay the first and second moment running average coefficient
                diff = grad - exp_avg
                exp_avg.mul_(beta1).add_(grad, alpha=(1. - beta1))
                exp_avg_var.mul_(beta2).addcmul_(diff, diff, value=1. - beta2) 
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_var.sqrt().add_(eps * sqrt_bias_correction2) 
                    # 'eps' is first multiplied by sqrt_bias_correction2 and then divided by sqrt_bias_correction2
                else:
                    denom = exp_avg_var.sqrt().add_(eps * sqrt_bias_correction2)

                p.data.addcdiv_(exp_avg, denom, value= - step_size * lr)

        return loss



class LaProp(Optimizer):
    def __init__(self, params, lr=4e-4, betas=(0.9, 0.999), eps=1e-15,
                 weight_decay=0., amsgrad=False, centered=False):
        self.centered = centered
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(LaProp, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LaProp, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if group['weight_decay'] != 0:
                    p.data.mul_(1. - group['lr'] * group['weight_decay'])

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LaProp does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of learning rates
                    state['exp_avg_lr_1'] = 0.
                    state['exp_avg_lr_2'] = 0.
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_mean_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    state['PopArt_rescale'] = 1.
                    state['Momentum_rescale'] = 1.

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if self.centered:
                    exp_mean_avg_sq = state['exp_mean_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                state['exp_avg_lr_1'] = state['exp_avg_lr_1'] * beta1 + (1. - beta1) * group['lr']
                state['exp_avg_lr_2'] = state['exp_avg_lr_2'] * beta2 + (1. - beta2)

                #bias_correction1 = state['exp_avg_lr_1'] / group['lr'] if group['lr']!=0. else 1. #1 - beta1 ** state['step']
                bias_correction2 = state['exp_avg_lr_2']

                # For convenience, we directly use "sqrt_bias_correction2" and "step_size" as the following
                sqrt_bias_correction2 = math.sqrt(bias_correction2)
                # when state['exp_avg_lr_1'] is zero, exp_avg should also be zero and it is trivial
                one_over_bias_correction1 = group['lr'] / state['exp_avg_lr_1'] if state['exp_avg_lr_1']!=0. else 0. 
                step_size = one_over_bias_correction1 
                
                # Decay the first and second moment running average coefficient
                rescaling = state['PopArt_rescale']
                exp_avg_sq.mul_(beta2*rescaling**2).addcmul_(grad, grad, value=1. - beta2)
                
                denom = exp_avg_sq
                if self.centered:
                    exp_mean_avg_sq.mul_(beta2*rescaling).add_(grad, alpha=1. - beta2)
                    if state['step']>5:
                        denom = denom.addcmul(exp_mean_avg_sq, exp_mean_avg_sq, value=-1.)

                if amsgrad:
                    if not (self.centered and state['step']<=5): 
                        # Maintains the maximum of all (centered) 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, denom, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq

                denom = denom.sqrt().add_(group['eps'] * sqrt_bias_correction2) # instead of correcting "denom" by dividing it, we put the correction factor into "exp_avg" and "eps"
                
                momentum_rescaling = state['Momentum_rescale'] 
                exp_avg.mul_(beta1*momentum_rescaling).addcdiv_(grad, denom, value=(1. - beta1) * group['lr'] * sqrt_bias_correction2)
                
                p.data.add_(exp_avg, alpha = -step_size)

        return loss


