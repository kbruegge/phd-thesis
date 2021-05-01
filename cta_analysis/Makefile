SHELL=/bin/bash
build_dir = ../build/cta_analysis/
data_dir = ../data/cta_data/

plot_overview = $(build_dir)separator_performance.pdf
plot_overview_regressor = $(build_dir)/regressor_performance.pdf

proton_test = $(build_dir)/proton_test.h5
gamma_test = $(build_dir)/gamma_test.h5
electron_test = $(build_dir)/electron_test.h5
proton_train = $(build_dir)/proton_train.h5
gamma_train = $(build_dir)/gamma_train.h5
gamma_pointlike = $(build_dir)/gamma_pointlike.h5

predictions_separator = $(build_dir)/aict_predictions_separation.h5
model_separator =  $(build_dir)/separator.pkl

predictions_regressor = $(build_dir)/aict_predictions_regression.h5
model_regressor = $(build_dir)/regressor.pkl

config = ../configs/aict/iact_config.yaml

#PLOTS := $(plot_overview) $(plot_overview_regressor)

all:  $(build_dir)/APPLICATION_DONE_BACKGROUND $(build_dir)/APPLICATION_DONE_SIGNAL 
#$(PLOTS)

clean:
	rm -rf $(build_dir)

$(build_dir):
	mkdir -p $(build_dir)

.INTERMEDIATE: psplit
$(proton_train): psplit
$(proton_test): psplit
psplit: $(data_dir)/protons.h5 | $(build_dir)
	# aict_apply_cuts $(config) $(data_dir)/protons.h5 $(build_dir)/protons.h5 -N 2000000
	aict_split_data $(data_dir)/protons.h5 $(build_dir)/proton -n train -f 0.015 -n test  -f 0.985  -t cta


.INTERMEDIATE: gsplit
$(gamma_train): gsplit
$(gamma_test): gsplit
gsplit: $(data_dir)/gammas_diffuse.h5 | $(build_dir)
	# aict_apply_cuts $(config) $(data_dir)/gammas_diffuse.h5 $(build_dir)/gammas_diffuse.h5 -N 2000000
	aict_split_data $(data_dir)/gammas_diffuse.h5 $(build_dir)/gamma -n train -f 0.05 -n test -f 0.95  -t cta
$(electron_test): $(data_dir)/electrons.h5 | $(build_dir)
	# aict_apply_cuts $(config) $(data_dir)/electrons.h5 $(electron_test) -N 2000000
	cp $(data_dir)/electrons.h5 $(electron_test)

$(gamma_pointlike): $(data_dir)/gammas.h5 | $(build_dir)
	cp $(data_dir)/gammas.h5 $(gamma_pointlike)
	# aict_apply_cuts $(config) $(data_dir)/gammas.h5 $(gamma_pointlike) -N 2000000



.INTERMEDIATE: clf
$(model_separator): clf
$(predictions_separator): clf
clf: $(proton_train) $(gamma_train) $(config)
	aict_train_separation_model $(config) $(gamma_train) $(proton_train) $(predictions_separator) $(model_separator)

.INTERMEDIATE: rgs
$(model_regressor): rgs
$(predictions_regressor): rgs
rgs: $(gamma_train) $(config)
	aict_train_energy_regressor $(config) $(gamma_train) $(predictions_regressor) $(model_regressor)

$(build_dir)/APPLICATION_DONE_BACKGROUND: $(model_separator) $(config) $(proton_test) $(model_regressor) $(electron_test)
	aict_apply_separation_model $(config) $(proton_test) $(model_separator) --yes --chunksize 1000000
	aict_apply_separation_model $(config) $(electron_test) $(model_separator) --yes --chunksize 100000

	aict_apply_energy_regressor $(config) $(proton_test) $(model_regressor) --yes --chunksize 1000000
	aict_apply_energy_regressor $(config) $(electron_test) $(model_regressor) --yes --chunksize 100000
	
	touch $(build_dir)/APPLICATION_DONE_BACKGROUND

$(build_dir)/APPLICATION_DONE_SIGNAL: $(model_separator) $(config) $(gamma_test) $(model_regressor) $(gamma_pointlike)
	aict_apply_separation_model $(config) $(gamma_test) $(model_separator) --yes --chunksize 400000
	aict_apply_separation_model $(config) $(gamma_pointlike) $(model_separator) --yes --chunksize 400000
	
	aict_apply_energy_regressor $(config) $(gamma_test) $(model_regressor) --yes --chunksize 400000
	aict_apply_energy_regressor $(config) $(gamma_pointlike) $(model_regressor) --yes --chunksize 400000
	
	touch $(build_dir)/APPLICATION_DONE_SIGNAL

# plot a bunch of performance values for the classifier
# $(plot_overview): $(model_separator) $(predictions_separator) $(config)
# 	aict_plot_separator_performance $(config) $(predictions_separator) $(model_separator) -o $(plot_overview)
# $(plot_overview_regressor): $(model_regressor) $(predictions_regressor)  $(config)
# 	aict_plot_regressor_performance $(config) $(predictions_regressor) $(model_regressor) -o $(plot_overview_regressor)

