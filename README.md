# llm-geneprioritization-framework

This project provides a framework for gene prioritization using LLM-based scoring, RAG verification, and interactive data exploration.

## Streamlit App

The Streamlit app (`app.py`) provides an interactive dashboard to:
- Explore gene data and prioritization results.
- Visualize cluster overlaps and candidate gene sets.
- View gene-wise details, scores, and justifications.
- Export gene lists for downstream analysis.

To run the app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notebooks

The included Jupyter notebooks demonstrate the pipeline and analysis steps:
- Data loading and preprocessing
- LLM-based scoring and RAG verification
- Visualization of results and candidate gene selection
- Download vector index :[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15802241.svg)](https://doi.org/10.5281/zenodo.15802241)

Example notebook: `3_runPipeLine.ipynb`

## Project Structure
- `app.py` — Main Streamlit dashboard
- `requirements.txt` — Python dependencies
- `data/` — Input data files
- `3_runPipeLine.ipynb` — Example analysis notebook

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Launch the Streamlit app: `streamlit run app.py`
3. Explore the notebooks for detailed analysis steps.

---

For questions or contributions, please open an issue or pull request.
