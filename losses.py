import torch
from torch.nn import functional as F

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float, kl_avg: str = "batchmean"):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.kl_avg = kl_avg

    
    def distillation_func(self, inputs, outputs_kd):
        
        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction=self.kl_avg,
                log_target=True
            ) * (T * T)
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        return distillation_loss

    def forward(self, inputs, outputs, labels, model=None):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        distillation_loss = self.distillation_func(inputs, outputs_kd)

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class DynamicViTDistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 ratio_weight: float, cls_distill_weight: float, token_distill_weight: float,
                 cls_weight: float, mse_token: bool, kl_avg: str = "batchmean"):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.cls_weight = cls_weight
        self.cls_distill_weight = cls_distill_weight
        self.token_distill_weight = token_distill_weight
        self.ratio_weight = ratio_weight
        self.mse_token = mse_token
        self.kl_avg = kl_avg

    def forward(self, inputs, outputs, labels, model):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        pred, token_pred, mask, out_pred_score = outputs

        
        base_loss = self.base_criterion(pred, labels)

        loss = base_loss * self.cls_weight

        pred_loss = 0.0
        try:
            keep_rate = model.module.token_ratio
        except AttributeError:
            keep_rate = model.token_ratio
            
        for i, score in enumerate(out_pred_score):
            pred_loss = pred_loss + ((score.mean(1) - keep_rate[i]) ** 2).mean()
        pred_loss /= len(out_pred_score)

        loss += pred_loss * self.ratio_weight


        if self.teacher_model is not None:
            with torch.no_grad():
                cls_t, token_t = self.teacher_model(inputs)

            cls_kl_loss = F.kl_div(
                    F.log_softmax(pred, dim=-1),
                    F.log_softmax(cls_t, dim=-1),
                    reduction=self.kl_avg,
                    log_target=True
                )

            loss += self.cls_distill_weight * cls_kl_loss

            B, N, C = token_pred.shape
            assert mask.numel() == B * N

            bool_mask = mask.reshape(B*N) > 0.5

            token_pred = token_pred.reshape(B*N, C)
            token_t = token_t.reshape(B*N, C)

            if mask.sum() < 0.1:
                token_kl_loss = token_pred.new(1,).fill_(0.0)
            else:
                token_t = token_t[bool_mask]
                token_pred = token_pred[bool_mask]
                if self.mse_token:
                    token_kl_loss = torch.pow(token_pred - token_t, 2).mean()
                else:
                    token_kl_loss = F.kl_div(
                            F.log_softmax(token_pred, dim=-1),
                            F.log_softmax(token_t, dim=-1),
                            reduction=self.kl_avg,
                            log_target=True
                        )
            
                loss += self.token_distill_weight * token_kl_loss

        return loss
