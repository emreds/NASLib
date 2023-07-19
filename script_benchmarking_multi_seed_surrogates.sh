#!/bin/bash
# Proper header for a Bash script.


python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_mlp_1.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_mlp_2.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_mlp_3.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_mlp_4.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_mlp_5.yaml

python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_xgb_1.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_xgb_2.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_xgb_3.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_xgb_4.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_xgb_5.yaml

python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_bananas_1.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_bananas_2.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_bananas_3.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_bananas_4.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_micro_vary_bananas_5.yaml


python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_MACs_vary_xgb_1.yaml
python naslib/runners/predictors/runner.py --config-file naslib/runners/predictors/predictor_config_lcz_42_latency_vary_xgb_1.yaml
