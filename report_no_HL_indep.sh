for f in main_out_new_no_HL_indep/out_*.csv; 
    do 
    echo $f; 
    python3 report_rmse_score.py CPC1.test_indep.json $f main_out_new_no_HL_indep/all_results.csv;
    python3 report_rmse_score_fitted.py CPC1.test_indep.json $f main_out_new_no_HL_indep/all_results_fitted.csv;
    python3 plot_test_indep.py CPC1.test_indep.json $f;
    done
