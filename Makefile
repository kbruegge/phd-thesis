TEXOPTS=--lualatex  --output-directory=build --interaction=nonstopmode -halt-on-error
TARGET=thesis

############## raw input data###############
gammas_input_full=data/cta_data/gammas.h5
gammas_diffuse_input_full=data/cta_data/gammas_diffuse.h5
protons_input_full=data/cta_data/protons.h5
electrons_input_full=data/cta_data/electrons.h5

#######################
# data files generated by the cta analysis makefile 
gamma_pointlike=build/cta_analysis/gamma_pointlike.h5
gamma_test=build/cta_analysis/gamma_test.h5
proton_test=build/cta_analysis/proton_test.h5
electron_test=build/cta_analysis/electron_test.h5
#######################

#######################
# pymc related plts for 
pymc_data=build/pymc_results/SPECTRA_DONE
pymc_fit=build/pymc_results/fit/FIT_DONE
pymc_unfold=build/pymc_results/unfold/UNFOLD_DONE
pymc_fit_results=build/pymc_results/pymc_fit_result.pgf
pymc_unfold_results=build/pymc_results/pymc_unfold_result.pdf
pymc_unfold_correlations=build/pymc_results/pymc_unfold_correlations.pdf
##########################

#########################################
# naima and model fit related things
model_variations=build/model_variations.pdf
naima_results_table=build/naima_results/density_0.pdf
naima_results_corner=build/naima_corner.pdf
sed_fit_he=build/sed_fit_he.pgf
ssc_fit=build/ssc_fit.pgf
##################################

4fgl=build/4fgl.pdf
erf=build/erf.pdf
boundary=build/boundary.pdf
error_propagation=build/error_propagation.pdf
funk=build/funk.pdf
wobble_mode=build/wobble_mode.pdf

array_layout=build/array_layout.pdf
array_layout_snippets=build/LAYOUT_SNIPPETS

preprocessing=build/preprocessing.pdf
preprocessing_snippet_energy=build/preprocessing_energy.txt
preprocessing_snippet_multi=build/preprocessing_multi.txt

hgps=build/hgps.pdf
cosmic_rays=build/cosmic_rays.pgf
sed_fit_he_txt=build/sed_fit_he.txt
sed_fit_he_matrix_txt=build/sed_fit_he_matrix.txt

fact_irf=build/fact_irf.pdf
iact_counts=build/iact_counts.pdf

ang_res_raw=build/ang_res_raw.pdf
ang_res_raw_pointlike=build/ang_res_raw_pointlike.pdf
ang_res_raw_mult=build/ang_res_raw_mult.pdf

hmax_raw=build/hmax_raw.pdf
impact_distance_raw=build/impact_distance_raw.pdf

rta=build/rta.pdf
############################################################
# things that dpeend on the cta_analysis result
energy_resolution_raw=build/energy_resolution_raw.pdf
energy_resolution_raw_csv=build/energy_resolution_raw.csv
energy_resolution_raw_e_reco=build/energy_resolution_raw_e_reco.pdf
energy_resolution_raw_e_reco_csv=build/energy_resolution_raw_e_reco.csv
energy_resolution_raw_e_reco_pointlike=build/energy_resolution_raw_e_reco_pointlike.pdf
energy_resolution_raw_e_reco_pointlike_csv=build/energy_resolution_raw_e_reco_pointlike.csv

importances_regressor=build/importances_regressor.pdf
auc_acc=build/auc_acc.pdf
importances_classifier=build/importances_classifier.pdf

theta_square=build/theta_square.pdf
theta_square_grid=build/theta_square_grid.pdf
sensitivity=build/sensitivity.pdf
sensitivity_csv=build/sensitivity.csv
sensitivity_fixed=build/sensitivity_fixed.pdf
sensitivity_fixed_csv=build/sensitivity_fixed.csv

event_selection=build/event_selection.txt

effective_area_optimized=build/effective_area_optimized.pdf
effective_area_optimized_fixed_theta=build/effective_area_optimized_fixed_theta.pdf
ang_res_optimized=build/ang_res_optimized.pdf
energy_resolution_optimized=build/energy_resolution_optimized.pdf
gamma_test_num_tel_events=build/gamma_test_num_tel_events.txt
gamma_test_num_array_events=build/gamma_test_num_array_events.txt
gamma_test_mean_multiplicity=build/gamma_test_mean_multiplicity.txt

classifier_snippets=build/classifier_k_cv.txt build/classifier_n_estimators.txt build/classifier_min_split.txt build/classifier_num_features.txt build/cv_auc.txt
regressor_snippets=build/regressor_k_cv.txt build/regressor_n_estimators.txt build/regressor_min_split.txt build/regressor_num_features.txt build/cv_r2.txt
cv_auc=build/cv_auc.txt
###################################################################


dataset_info=build/dataset_info.txt
cleaning_info=build/cleaning_info.txt
dl2_info_tel=build/dl2_info_telescope.txt
dl2_info_runs=build/dl2_info_runs.txt
dl2_info_array=build/dl2_info_array.txt


len_train_protons=build/len_train_proton.txt
len_train_gammas=build/len_train_gamma.txt
len_test_protons=buildlen_test_proton.txt
len_test_gammas=build/len_test_gamma.txt

theta_square_rate_snippet=build/theta_square_rate_raw.txt



all: $(TARGET).pdf

PLOTS := $(4fgl) $(erf) $(hgps) $(sed_fit_he) $(cosmic_rays) $(model_variations) $(funk) $(ssc_fit) $(naima_results_table) $(error_propagation) 
PLOTS += $(fact_irf) $(wobble_mode) $(iact_counts) $(pymc_fit_results) $(pymc_unfold_results) $(pymc_unfold_correlations) $(preprocessing) $(ang_res_raw) $(ang_res_raw_mult) $(ang_res_raw_pointlike)
PLOTS += $(array_layout) $(hmax_raw) $(impact_distance_raw) $(boundary) $(auc_acc) $(importances_classifier) $(importances_regressor)
PLOTS += $(naima_results_corner) $(theta_square)  $(energy_resolution_raw) $(energy_resolution_raw_e_reco) $(energy_resolution_raw_e_reco_pointlike)
PLOTS += $(sensitivity) $(sensitivity_fixed) $(ang_res_optimized) $(effective_area_optimized) $(effective_area_optimized_fixed_theta) $(energy_resolution_optimized) $(rta) $(theta_square_grid) 

SNIPPETS := $(sed_fit_he_txt) $(sed_fit_he_matrix_txt) $(preprocessing_snippet_energy) $(preprocessing_snippet_multi)
SNIPPETS += $(gamma_test_num_tel_events) $(gamma_test_num_array_events) $(gamma_test_mean_multiplicity) $(array_layout_snippets) $(dataset_info) $(cleaning_info)
SNIPPETS += $(dl2_info_array) $(dl2_info_tel) $(dl2_info_runs)  $(classifier_snippets) $(regressor_snippets)
SNIPPETS += $(len_train_gammas) $(len_train_protons) $(cv_auc) $(event_selection) build/requirements.txt $(theta_square_rate_raw)



build/titlepage.pdf: titlepage.tex  | build
	latexmk $(TEXOPTS) titlepage.tex

build/praegung.pdf: praegung.tex  | build
	latexmk $(TEXOPTS) praegung.tex

build/ruecken.pdf: ruecken.tex  | build
	latexmk $(TEXOPTS) ruecken.tex

build/abstract.pdf: abstract_standalone.tex  | build
	latexmk $(TEXOPTS) abstract_standalone.tex

$(TARGET).pdf: $(TARGET).tex mimosis.cls $(PLOTS) $(FIGURES) $(SNIPPETS) build/titlepage.pdf build/abstract.pdf | build
	latexmk $(TEXOPTS) $(TARGET).tex

preview: $(TARGET).tex $(PLOTS) $(FIGURES) build/titlepage.pdf | build
	latexmk $(TEXOPTS) -pvc $(TARGET).tex

build:
	mkdir build
	mkdir build/naima_results
	mkdir build/pymc_results

clean:
	rm -r build

$(pymc_data): ./configs/pymc/data_conf.yaml | build
	which pymc_create_observations
	pymc_create_observations ./plots/data/joint_crab/dl3/ ./configs/pymc/data_conf.yaml build/pymc_results/spectra
	touch $(pymc_data)

$(pymc_fit): $(pymc_data) | build
	pymc_fit_spectrum build/pymc_results/spectra build/pymc_results/fit/magic magic --model_type full  --n_tune 2000 --n_samples 10000 
	pymc_fit_spectrum build/pymc_results/spectra build/pymc_results/fit/veritas veritas --model_type full  --n_tune 2000 --n_samples 10000
	pymc_fit_spectrum build/pymc_results/spectra build/pymc_results/fit/hess hess --model_type full  --n_tune 2000 --n_samples 10000
	pymc_fit_spectrum build/pymc_results/spectra build/pymc_results/fit/fact fact --model_type full  --n_tune 2000 --n_samples 10000
	touch $(pymc_fit)

$(pymc_unfold):  ./configs/pymc/data_conf_unfold.yaml  | build
	pymc_unfold_spectrum ./plots/data/joint_crab/dl3 ./configs/pymc/data_conf_unfold.yaml build/pymc_results/unfold/hess hess --n_tune=1500 --n_samples=3000 
	pymc_unfold_spectrum ./plots/data/joint_crab/dl3 ./configs/pymc/data_conf_unfold.yaml build/pymc_results/unfold/veritas veritas --n_tune=1500 --n_samples=3000 
	pymc_unfold_spectrum ./plots/data/joint_crab/dl3 ./configs/pymc/data_conf_unfold.yaml build/pymc_results/unfold/magic magic --n_tune=1500 --n_samples=3000 
	pymc_unfold_spectrum ./plots/data/joint_crab/dl3 ./configs/pymc/data_conf_unfold.yaml build/pymc_results/unfold/fact fact --n_tune=1500 --n_samples=3000 
	touch $(pymc_unfold)

$(pymc_fit_results): $(pymc_fit) $(pymc_unfold) plots/pymc_fit_results.py matplotlibrc | build
	python plots/pymc_fit_results.py

$(pymc_unfold_results): $(pymc_unfold) plots/pymc_unfold_results.py matplotlibrc | build
	python plots/pymc_unfold_results.py

$(pymc_unfold_correlations): $(pymc_unfold) plots/pymc_unfold_correlations.py  matplotlibrc | build	
	python plots/pymc_unfold_correlations.py

$(4fgl): plots/fermi_plot.py matplotlibrc | build
	python plots/fermi_plot.py

$(hgps): plots/hess_gps.py matplotlibrc | build
	python plots/hess_gps.py

$(erf): plots/erf.py matplotlibrc | build
	python plots/erf.py

$(boundary): plots/boundary.py matplotlibrc | build
	python plots/boundary.py

$(error_propagation): plots/error_propagation.py matplotlibrc | build
	python plots/error_propagation.py

$(cosmic_rays): plots/cosmic_rays.py matplotlibrc | build
	python plots/cosmic_rays.py

$(fact_irf): plots/fact_irf.py matplotlibrc | build
	python plots/fact_irf.py

$(wobble_mode): plots/wobble_mode.py matplotlibrc | build
	python plots/wobble_mode.py

$(model_variations): plots/model_variations.py matplotlibrc | build
	python plots/model_variations.py

$(iact_counts): plots/iact_counts.py plots/iact_overview.py $(pymc_data) matplotlibrc | build
	python plots/iact_counts.py
	python plots/iact_overview.py

.INTERMEDIATE: sed
$(sed_fit_he):sed
$(sed_fit_he_txt):sed
$(sed_fit_he_matrix_txt):sed
sed: plots/sed_fit_he.py matplotlibrc | build
	python plots/sed_fit_he.py

$(ssc_fit): plots/naima_sed_plot.py matplotlibrc | build
	python plots/naima_sed_plot.py

# force multiple outputs to not be created in parallel multiple times
# https://stackoverflow.com/a/47951465/2154625
.INTERMEDIATE: d
$(naima_results_table):d
$(naima_results_corner):d
d: plots/naima_parameter.py $(ssc_fit) | build
	python plots/naima_parameter.py 

$(funk): plots/funk.py matplotlibrc | build
	python plots/funk.py

.INTERMEDIATE: prepro
$(preprocessing): prepro
$(preprocessing_snippet_energy): prepro
$(preprocessing_snippet_multi): prepro
prepro: plots/preprocessing.py matplotlibrc | build
	python plots/preprocessing.py


$(array_layout) $(array_layout_snippets): plots/array_layout.py matplotlibrc | build
	python plots/array_layout.py
	touch $(array_layout_snippets)

.INTERMEDIATE: ang
$(ang_res_raw):ang
$(ang_res_raw_data):ang
$(ang_res_raw_mult):ang
ang: matplotlibrc | $(gamma_test) build
	cta_plot_reco -o $(ang_res_raw) --no-ylog $(gamma_test) angular-resolution
	cta_plot_reco -o $(ang_res_raw_mult) --no-ylog $(gamma_test) angular-resolution-multiplicity 

$(ang_res_raw_pointlike) $(ang_res_raw_data_pointlike): matplotlibrc $(gamma_pointlike) | build
	cta_plot_reco -o $(ang_res_raw_pointlike) --no-ylog $(gamma_pointlike) angular-resolution

$(hmax_raw): matplotlibrc | build
	cta_plot_reco -o $(hmax_raw) --no-ylog data/cta_data/gammas_diffuse.h5 h-max
$(impact_distance_raw): matplotlibrc | build
	cta_plot_reco -o $(impact_distance_raw) --no-ylog data/cta_data/gammas_diffuse.h5 impact-distance

.INTERMEDIATE: dinf
$(gamma_test_num_tel_events):dinf
$(gamma_test_num_array_events):dinf
$(gamma_test_mean_multiplicity):dinf
$(dataset_info): dinf
dinf: plots/dataset_info.py data/cta_data/gammas_diffuse.h5
	python plots/dataset_info.py

.INTERMEDIATE: inf
$(len_train_protons):inf 
$(len_train_gammas):inf
$(len_test_protons): inf
$(len_test_gammas):inf 
$(classifier_snippets): inf
$(regressor_snippets): inf
inf: plots/ml_data_info.py build/cta_analysis/regressor.pkl build/cta_analysis/separator.pkl 
	python plots/ml_data_info.py

.INTERMEDIATE: dl2inf
$(dl2_info_runs):dl2inf
$(dl2_info_array):dl2inf
$(dl2_info_tel):dl2inf
dl2inf: plots/dl2_info.py
	python plots/dl2_info.py

$(cleaning_info): plots/cleaning_info.py configs/preprocessing/config.yaml
	python plots/cleaning_info.py

.INTERMEDIATE: c
$(proton_test):c
$(gamma_test):c
$(gamma_pointlike):c
$(electron_test):c
build/cta_analysis/separator.pkl:c
build/cta_analysis/regressor.pkl:c
build/cta_analysis/APPLICATION_DONE_SIGNAL:c
build/cta_analysis/APPLICATION_DONE_BACKGROUND:c
c: configs/aict/iact_config.yaml $(gammas_input_full) $(protons_input_full) $(electrons_input_full) $(gammas_diffuse_input_full) |build
	cd cta_analysis; make 

$(energy_resolution_raw) $(energy_resolution_raw_csv): matplotlibrc $(gamma_test) build/cta_analysis/regressor.pkl | build
	cta_plot_reco --ylim -0.1 1  -o $(energy_resolution_raw) $(gamma_test) energy-resolution 
$(energy_resolution_raw_e_reco) $(energy_resolution_raw_e_reco_csv): matplotlibrc $(gamma_test) build/cta_analysis/regressor.pkl | build
	cta_plot_reco -o $(energy_resolution_raw_e_reco) $(gamma_test) energy-resolution --plot_e_reco
$(energy_resolution_raw_e_reco_pointlike) $(energy_resolution_raw_e_reco_pointlike_csv): matplotlibrc $(gamma_pointlike) build/cta_analysis/regressor.pkl | build
	cta_plot_reco -o $(energy_resolution_raw_e_reco_pointlike) $(gamma_pointlike) energy-resolution  --plot_e_reco

$(auc_acc): matplotlibrc $(gamma_test) $(proton_test) build/cta_analysis/separator.pkl build/cta_analysis/APPLICATION_DONE_SIGNAL | build
	cta_plot_classifier -o $(auc_acc) $(gamma_test) $(proton_test)  roc-acc 
$(importances_classifier): matplotlibrc build/cta_analysis/separator.pkl | build
	cta_plot_importance  build/cta_analysis/separator.pkl -o $(importances_classifier) --xlim 0 0.26
$(importances_regressor): matplotlibrc build/cta_analysis/regressor.pkl | build
	cta_plot_importance build/cta_analysis/regressor.pkl --color "#5f218c" -o $(importances_regressor)  

.INTERMEDIATE: eff
$(effective_area_optimized):eff
$(effective_area_optimized_fixed_theta):eff
eff: matplotlibrc $(gamma_pointlike) $(sensitivity_csv) $(sensitivity_fixed_csv) | build
	cta_plot_effective_area $(gamma_pointlike) -o $(effective_area_optimized) --cuts_path $(sensitivity_csv)
	cta_plot_effective_area $(gamma_pointlike) -o $(effective_area_optimized_fixed_theta) --cuts_path $(sensitivity_fixed_csv) --cmap viridis

$(ang_res_optimized): matplotlibrc $(gamma_pointlike) $(sensitivity_csv) | build
	cta_plot_reco -o $(ang_res_optimized) --cuts_path $(sensitivity_csv) --ylim 0 0.52 --no-ylog $(gamma_pointlike) angular-resolution --reference --plot_e_reco
$(energy_resolution_optimized): matplotlibrc $(gamma_pointlike) $(sensitivity_csv) | build
	cta_plot_reco --cuts_path $(sensitivity_csv) -o $(energy_resolution_optimized) --ylim -0.1 1.0 $(gamma_pointlike) energy-resolution  --plot_e_reco --reference

.INTERMEDIATE: theta
$(theta_square):theta
$(theta_square_rate_snippet):theta
theta: matplotlibrc $(gamma_pointlike) $(proton_test) $(electron_test) build/cta_analysis/APPLICATION_DONE_SIGNAL build/cta_analysis/APPLICATION_DONE_BACKGROUND | build
	cta_plot_theta_square $(gamma_pointlike) $(proton_test) $(electron_test) -o $(theta_square)

$(theta_square_grid): matplotlibrc $(gamma_pointlike) $(proton_test) $(electron_test) build/cta_analysis/APPLICATION_DONE_SIGNAL build/cta_analysis/APPLICATION_DONE_BACKGROUND | build
	cta_plot_theta_square_grid $(gamma_pointlike) $(proton_test) $(electron_test) -o $(theta_square_grid) 
# force multiple outputs to not be created in parallel multiple times
# https://stackoverflow.com/a/47951465/2154625
.INTERMEDIATE: b
$(sensitivity):b
$(sensitivity_csv):b
$(sensitivity_fixed):b
$(sensitivity_fixed_csv):b
b: matplotlibrc $(gamma_pointlike) $(proton_test) $(electron_test) build/cta_analysis/APPLICATION_DONE_SIGNAL build/cta_analysis/APPLICATION_DONE_BACKGROUND | $(build_dir)
	cta_plot_sensitivity $(gamma_pointlike) $(proton_test) $(electron_test) -o $(sensitivity) --reference --requirement 
	cta_plot_sensitivity $(gamma_pointlike) $(proton_test) $(electron_test) -o $(sensitivity_fixed) --reference --requirement --fix_theta --color "xkcd:green"


$(event_selection): $(sensitivity_csv) plots/event_selection.py | build
	python plots/event_selection.py

$(rta): data/rta/output.csv.gz plots/rta.py | build
	python plots/rta.py

build/requirements.txt:
	pip freeze > build/requirements.txt