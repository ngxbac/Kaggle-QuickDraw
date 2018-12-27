import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from catalyst.utils.factory import UtilsFactory
from catalyst.dl.callbacks import (
    ClassificationLossCallback, Callback, InferCallback,
    BaseMetrics, Logger, TensorboardLogger,
    OptimizerCallback, CheckpointCallback,
    PrecisionCallback, OneCycleLR, LRFinder, MapKCallback,
    SchedulerCallback)
from catalyst.dl.runner import AbstractModelRunner
from catalyst.modules.pooling import GlobalConcatPool1d
from catalyst.dl.state import RunnerState
from cnn_finetune import make_model


# ---- Model ----

def make_classifier(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, num_classes),
    )


class BaselineModel(nn.Module):
    def __init__(self,
                 arch="resnet18",
                 n_class=10,
                 pretrained=True,
                 num_embeddings=750,
                 embedding_dim=10):
        super(BaselineModel, self).__init__()
        resnet = make_model(
            model_name=arch,
            num_classes=n_class,
            pretrained=pretrained,
        )
        print(resnet)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            resnet._features[1],
        )

        self.encoder3 = resnet._features[2]
        self.encoder4 = resnet._features[3]
        self.encoder5 = resnet._features[4]

        # Embedding layers
        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_embeddings+1, embedding_dim=embedding_dim),
            nn.Dropout(0.25),
            GlobalConcatPool1d(),
        )

        self.logit = nn.Linear(2068, n_class)

    def forward(self, x, countrycode):
        # Image features
        batch_size, C, H, W = x.shape
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.encoder5(x)

        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size,-1)
        x = F.dropout(x, p=0.50, training=self.training)

        # Embedding features
        embedding = self.embedding(countrycode)
        # print("Embedding shape ", embedding.size())
        embedding = embedding.view(embedding.size(0), -1)

        # Concat all
        x = torch.cat([x, embedding], 1)
        # Pass to logit layer
        logit = self.logit(x)
        return logit


class Finetune(nn.Module):
    def __init__(self,
                 arch="resnet18",
                 n_class=10,
                 pretrained=True,
                 num_embeddings=750,
                 embedding_dim=10):
        super(Finetune, self).__init__()
        self.model = make_model(
            model_name=arch,
            num_classes=n_class,
            pretrained=pretrained,
        )

        # Embedding layers
        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_embeddings+1, embedding_dim=embedding_dim),
            nn.Dropout(0.25),
            GlobalConcatPool1d(),
        )

        # Exception: 2068
        # Densenet161: 2228
        # dpn68: 852
        # resnet34: 532
        # n_image_features = self.model._classifier.in_features
        # n_embedding_features = embedding_dim * 2
        # self.logit = nn.Linear(n_embedding_features + n_image_features, n_class)
        self.logit = nn.Linear(532, n_class)

    def forward(self, x, countrycode):
        x = self.model._features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        # Embedding features
        embedding = self.embedding(countrycode)
        # print("Embedding shape ", embedding.size())
        embedding = embedding.view(embedding.size(0), -1)

        # Concat all
        x = torch.cat([x, embedding], 1)
        # Pass to logit layer
        logit = self.logit(x)
        return logit


class FinetuneImage(nn.Module):
    def __init__(self,
                 arch="se_resnext101_32x4d",
                 num_classes=340,
                 pretrained=True
        ):
        super(FinetuneImage, self).__init__()
        self.base_model = make_model(
            model_name=arch,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    def forward(self, x):
        return self.base_model(x)


class CleanLabelModel(nn.Module):
    def __init__(self,
                 arch="resnet34",
                 num_classes=340,
                 pretrained=True
    ):
        super(CleanLabelModel, self).__init__()
        resnet = make_model(
            model_name=arch,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        # print(resnet)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            resnet._features[4],
        )

        self.encoder3 = resnet._features[5]
        self.encoder4 = resnet._features[6]
        self.encoder5 = resnet._features[7]

        self.logit = nn.Linear(512, num_classes)

    def forward(self, x):
        x_cnn = x
        x_cnn = self.encoder1(x_cnn)
        x_cnn = self.encoder2(x_cnn)
        x_cnn = self.encoder3(x_cnn)
        x_cnn = self.encoder4(x_cnn)
        x_cnn = self.encoder5(x_cnn)
        # x_cnn = self.model._features(x_cnn)
        x_cnn = F.adaptive_avg_pool2d(x_cnn, 1)
        x_cnn = x_cnn.view(x.size(0), -1)

        logit = self.logit(x_cnn)
        return logit


class RNN(nn.Module):
    def __init__(self,
                 sequence_length,
                 input_size,
                 hidden_size,
                 num_layers,
                 num_embeddings,
                 embedding_dim,
                 num_classes,
                 use_cnn,
                 use_embedding,
                 use_lstm_image,
                 use_lstm_stroke,
    ):
        super(RNN, self).__init__()
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_cnn = use_cnn
        self.use_embedding = use_embedding
        self.use_lstm_image = use_lstm_image
        self.use_lstm_stroke = use_lstm_stroke

        output_num = 0

        if self.use_lstm_image:
            # LSTM with image input
            self.lstm_image = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            output_num += hidden_size

        if self.use_lstm_stroke:
            """
            WaveNet
            """
            from wavenet.networks import WaveNet
            self.bnwavenet = nn.BatchNorm1d(3)
            self.wavenet = WaveNet(
                layer_size=3,
                stack_size=5,
                in_channels=3,
                res_channels=32
            )
            print(self.wavenet)
            # LSTM with stroke input
            self.conv1d_stroke = nn.Sequential(
                nn.BatchNorm1d(3),
                nn.Conv1d(in_channels=3, out_channels=48, kernel_size=5, padding=0, stride=1, bias=False),
                nn.Dropout(0.3),
                nn.Conv1d(in_channels=48, out_channels=64, kernel_size=5, padding=0, stride=1, bias=False),
                nn.Dropout(0.3),
                nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, padding=0, stride=1, bias=False),
                nn.Dropout(0.3),
            )

            self.lstm_stroke_1 = nn.LSTM(input_size=96, hidden_size=128, num_layers=1, batch_first=True)
            self.lstm_stroke_1.flatten_parameters()

            self.lstm_stroke_2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
            self.lstm_stroke_2.flatten_parameters()

            output_num += 128 + 45

        if self.use_cnn:
            resnet = make_model(
                model_name="resnet34",
                num_classes=num_classes,
                pretrained=True,
            )
            # print(resnet)

            self.encoder1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

            self.encoder2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                resnet._features[4],
            )

            self.encoder3 = resnet._features[5]
            self.encoder4 = resnet._features[6]
            self.encoder5 = resnet._features[7]

            output_num += 512

        if self.use_embedding:
            # Embedding layers
            self.embedding = nn.Sequential(
                nn.Embedding(num_embeddings=num_embeddings+1, embedding_dim=embedding_dim),
                nn.Dropout(0.25),
                GlobalConcatPool1d(),
            )

            output_num += embedding_dim * 2

        self.bn1d = nn.BatchNorm1d(output_num)
        self.fc = nn.Linear(output_num, num_classes)

    def forward(self, x, x_gray, x_stroke, countrycode):
        # print(self.use_cnn)
        # print(self.use_embedding)
        # print(self.use_lstm_image)
        # print(self.use_lstm_stroke)

        output = []
        if self.use_cnn:
            # x_cnn = torch.cat((x, x, x), 1)
            x_cnn = x
            x_cnn = self.encoder1(x_cnn)
            x_cnn = self.encoder2(x_cnn)
            x_cnn = self.encoder3(x_cnn)
            x_cnn = self.encoder4(x_cnn)
            x_cnn = self.encoder5(x_cnn)
            # x_cnn = self.model._features(x_cnn)
            x_cnn = F.adaptive_avg_pool2d(x_cnn, 1)
            x_cnn = x_cnn.view(x.size(0), -1)
            output.append(x_cnn)

        if self.use_lstm_image:
            x_lstm_image_input = x_gray.reshape(-1, self.sequence_length, self.input_size)

            # Set initial hidden and cell states
            h0 = torch.zeros(self.num_layers, x_gray.size(0), self.hidden_size).cuda()
            c0 = torch.zeros(self.num_layers, x_gray.size(0), self.hidden_size).cuda()

            # Forward propagate LSTM
            self.lstm_image.flatten_parameters()
            x_lstm_image, _ = self.lstm_image(x_lstm_image_input, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
            x_lstm_image = x_lstm_image[:, -1, :]
            output.append(x_lstm_image)

        if self.use_lstm_stroke:
            x_wavenet = x_stroke
            x_wavenet = self.bnwavenet(x_wavenet)
            wave_out = self.wavenet(x_wavenet.transpose(1, 2))
            wave_out = wave_out.view(wave_out.size(0), -1)
            # print("WAVE OUT ", wave_out.shape)
            # LSTM with stroke
            # print("INPUT STROKE SIZE ", x_stroke.shape)
            x_lstm_stroke = self.conv1d_stroke(x_stroke)
            x_lstm_stroke = x_lstm_stroke.permute(0, 2, 1)
            # print("LSTM STROKE 1 ", x_lstm_stroke.shape)

            # Set initial hidden and cell states
            h0_stroke = torch.zeros(1, x_stroke.size(0), self.hidden_size).cuda()
            c0_stroke = torch.zeros(1, x_stroke.size(0), self.hidden_size).cuda()
            self.lstm_stroke_1.flatten_parameters()
            x_lstm_stroke, _ = self.lstm_stroke_1(x_lstm_stroke, (h0_stroke, c0_stroke))
            # print("LSTM STROKE 2 ", x_lstm_stroke.shape)

            h0_stroke = torch.zeros(1, x_stroke.size(0), self.hidden_size).cuda()
            c0_stroke = torch.zeros(1, x_stroke.size(0), self.hidden_size).cuda()
            self.lstm_stroke_2.flatten_parameters()
            x_lstm_stroke, _ = self.lstm_stroke_2(x_lstm_stroke, (h0_stroke, c0_stroke))
            # print("LSTM STROKE 3 ", x_lstm_stroke.shape)
            x_lstm_stroke = x_lstm_stroke[:, -1, :]

            output.append(x_lstm_stroke)
            output.append(wave_out)

        if self.use_embedding:
            # Embedding features
            embedding = self.embedding(countrycode)
            embedding = embedding.view(embedding.size(0), -1)
            output.append(embedding)

        n_output = len(output)
        output = torch.cat(output, 1)

        if n_output > 1:
            output = self.bn1d(output)
        output = self.fc(output)
        return output


def build_baseline_model(img_encoder):
    net = BaselineModel(**img_encoder)
    return net


def build_finetune_model(img_encoder):
    net = Finetune(**img_encoder)
    return net


def build_lstm_model(img_encoder):
    net = RNN(**img_encoder)
    return net


def build_clean_mode(img_encoder):
    net = CleanLabelModel(**img_encoder)
    return net


def build_finetune_image(img_encoder):
    return FinetuneImage(**img_encoder)


NETWORKS = {
    "baseline": build_baseline_model,
    "finetune": build_finetune_model,
    "lstmbased": build_lstm_model,
    "cleanmodel1": build_clean_mode,
    "FinetuneImage": build_finetune_image,
}


def prepare_model(config):
    return UtilsFactory.create_model(
        config=config, available_networks=NETWORKS)


def prepare_logdir(config):
    model_params = config["model_params"]
    return f"{model_params['model']}" \
           f"-{model_params['img_encoder']['arch']}" \
           f"-{model_params['img_encoder']['pooling']}"


# ---- Callbacks ----

class StageCallback(Callback):
    def on_stage_init(self, model, stage):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage in ["debug", "stage1"]:
            pass
        elif stage == "stage2":
            pass
        else:
            raise NotImplemented


class LossCallback(Callback):
    def __init__(self, emb_l2_reg=-1):
        self.emb_l2_reg = emb_l2_reg
        self.class_weight = np.load("class_weight.npy")
        self.class_weight = torch.from_numpy(self.class_weight).float().cuda()

    def on_batch_end(self, state):
        logits = state.output["logits"]
        # loss = state.criterion(logits.float(), state.input["targets"].long())
        loss = F.cross_entropy(logits.float(), state.input["targets"].long(), weight=self.class_weight)
        state.loss["main"] = loss
        del logits
        del loss


# ---- Runner ----

class ModelRunner(AbstractModelRunner):

    def _init_state(
            self, *,
            mode: str,
            stage: str = None,
            **kwargs) -> RunnerState:
        """
        Inner method for children's classes for state specific initialization.
        :return: RunnerState with all necessary parameters.
        """
        additional_kwargs = {}

        if mode == "train":
            additional_kwargs["criterion"] = self.criterion.get("main", None)
            additional_kwargs["main_metric"] = "map03"
            additional_kwargs["minimize_metric"] = False

        return super()._init_state(mode=mode, stage=stage, **additional_kwargs)

    @staticmethod
    def prepare_callbacks(
            *,
            args,
            mode,
            stage=None,
            **kwargs
    ):
        callbacks = collections.OrderedDict()

        if mode == "train":
            if stage == "debug":
                callbacks["stage"] = StageCallback()
                callbacks["loss"] = LossCallback(
                    emb_l2_reg=kwargs.get("emb_l2_reg", -1))
                callbacks["optimizer"] = OptimizerCallback(
                    grad_clip=kwargs.get("grad_clip", None))
                callbacks["metrics"] = BaseMetrics()
                callbacks["lr-finder"] = LRFinder(
                    final_lr=kwargs.get("final_lr", 0.1),
                    n_steps=kwargs.get("n_steps", None))
                callbacks["logger"] = Logger()
                callbacks["tflogger"] = TensorboardLogger()
            else:
                callbacks["stage"] = StageCallback()
                callbacks["loss"] = LossCallback(
                    emb_l2_reg=kwargs.get("emb_l2_reg", -1))
                callbacks["optimizer"] = OptimizerCallback(
                    grad_clip=kwargs.get("grad_clip", None))
                callbacks["metrics"] = BaseMetrics()
                callbacks["map"] = MapKCallback(
                    map_args=kwargs.get(
                        "map_args", [3]))
                callbacks["saver"] = CheckpointCallback(
                    save_n_best=getattr(args, "save_n_best", 7),
                    resume=args.resume
                )

                # Pytorch scheduler callback
                callbacks["scheduler"] = SchedulerCallback(
                    reduce_metric="map03")

                callbacks["logger"] = Logger()
                callbacks["tflogger"] = TensorboardLogger()
        elif mode == "infer":
            callbacks["saver"] = CheckpointCallback(resume=args.resume)
            callbacks["infer"] = InferCallback(out_prefix=args.out_prefix)
        else:
            raise NotImplementedError

        return callbacks

    @staticmethod
    def _batch_handler(*, dct, model):
        logits = model(dct["image"])
        output = {"logits": logits}
        return output
