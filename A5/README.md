first run `parse_to_rs3.py` and ensure that `paragraph.txt` is in the same directory.
    this will generate 2 files for each paragraph, one is the rs3 tree and the other is a png of the same
then run `extract_rst_info.py`. this step requires the previous one to be done already. it needs *.rs3 and *.png
    this will generate a json of the extracted info telling about the relation types and nucleus/satellite labels.
then run `rst_summarizer.py` and ensure the prev json is available in the same directory.
    this will finally give us `rst_summarization_report.txt` which i have modified manually to include the summaries given to me by `Claude Sonnet 4.5`
then run `metrics.py`, the output will be in `metrics_report.txt`

actually, just run `pipeline.py`