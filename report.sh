for f in save/*_preds.csv; 
    do 
    echo $f; 
    python3 report_rmse_score.py $f --use_CPC1
    python3 report_rmse_score.py $f --use_CPC1 --use_fitted
    # python3 report_rmse_score_fitted.py CPC1.test.json $f save/all_results_fitted.csv;
    # python3 plot_test.py CPC1.test.json $f;
    done
