import streamlit as st
import pickle
import os
import pandas as pd

# Try to import plotly_events and set a flag
try:
    from streamlit_plotly_events import plotly_events
    plotly_events_available = True
except ImportError:
    plotly_events_available = False

# load data and summary table files
# Path to the pickle file
data_dir = './data'
pkl_file = 'PS1_rago4_phi4_ss.pkl'
pkl_path = os.path.join(data_dir, pkl_file)

def load_data(path):
    return pd.read_pickle(path)

df = load_data(pkl_path)
# read the summary data
stable = pd.read_csv(os.path.join(data_dir, 'PS1_to_Candidate_Sepsis_genes.csv'), index_col=0)

st.set_page_config(page_title='Gene Data Explorer', layout='wide')

# Try to infer gene name column
gene_col = None
for col in df.columns:
    if 'gene' in col.lower():
        gene_col = col
        break
if gene_col is None:
    gene_col = st.selectbox('Select gene name column', df.columns)

# Try to infer json_key column
json_key_col = None
for col in df.columns:
    if 'json' in col.lower() or 'key' in col.lower():
        json_key_col = col
        break
if json_key_col is None:
    json_key_col = st.selectbox('Select json_key column', [col for col in df.columns if col != gene_col])

# Sidebar for page selection
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Overview', 'Gene-wise Detail','Resources'])

# Sidebar for gene and json_key selection (for detail page)
gene_names = df[gene_col].unique()
json_keys = df[json_key_col].unique()

if page == 'Overview':
    # show method overview figure
    ##############################################################################
    st.subheader('Candidate Gene Selection Workflow')
    # show method overview figure
    # Note: The image path should be relative to the app.py file location
    st.image('Figures/Figure1_CandidateGeneSelection_workflow.png',use_container_width=True)
    st.markdown('''
This workflow outlines the process of selecting candidate genes for sepsis based on their evaluation through Naive LLM and Hybrid LLM methods. The steps include:
1. **Naive LLM Screening**: Genes are initially screened using Naive LLM, where a score of ≥5 in at least one of the eight criteria indicates potential relevance to sepsis.
2. **Hybrid LLM Evaluation**: The screened genes undergo further evaluation using Hybrid LLM, which combines multiple criteria to refine the selection.
3. **Priority Clustering**: Genes are clustered based on their scores from both Naive LLM and Hybrid LLM evaluations. This clustering helps in identifying genes with similar characteristics and potential relevance to sepsis.
4. **Candidate Gene Selection**: The final candidate genes are selected based on their clustering results. Genes in the top cluster (Priority Cluster 1) are considered for further validation and experimental studies.
''')
    st.markdown('''
**Note**: The workflow is designed to ensure a systematic approach to gene selection, minimizing false positives and focusing on genes with the highest potential relevance to sepsis.
''')
    ##############################################################################

    st.title('Gene Data Overview')
    # show the summary table
    st.subheader('Summary Table for Priority Set 1 (PS1) Genes')
    st.dataframe(stable[['HybridLLM_cluster', 'NaiveLLM_cluster', 'priority_cluster','NaiveLLM_score', 'HybridLLM_score']])
    # show the snakey plot
    st.subheader('Cluster Overlap: NaiveLLM vs Hybrid LLM (Sankey Plot)')
    # Sankey plot requires plotly
    import plotly.graph_objects as go

    # Define your color mapping for clusters
    cl_Cat = {'5': '#1f78b4',
    '4': '#a6cee3',
    '3': '#b2df8a',
    '2': '#f4a582',
    '1': '#ca0020',
    'Fail': '#0b0b0b'}


    import random
    genes = stable.index.tolist()
    naive_clusters = pd.Series(stable['NaiveLLM_cluster'].astype(str),  index=stable.index)
    hybrid_clusters = pd.Series(stable['HybridLLM_cluster'].astype(str),  index=stable.index)

    # Build a DataFrame for mapping
    df_sankey = pd.DataFrame({
        'gene': stable.index,
        'naive_cluster': naive_clusters,
        'hybrid_cluster': hybrid_clusters
    })

    # Count overlaps
    overlap_counts = df_sankey.groupby(['naive_cluster', 'hybrid_cluster']).size().reset_index(name='count')

    # Prepare Sankey node labels
    naive_labels = sorted(df_sankey['naive_cluster'].unique())
    hybrid_labels = sorted(df_sankey['hybrid_cluster'].unique())
    labels = naive_labels + hybrid_labels

    # Map cluster names to node indices
    naive_idx = {k: i for i, k in enumerate(naive_labels)}
    hybrid_idx = {k: i+len(naive_labels) for i, k in enumerate(hybrid_labels)}

    # Build source, target, value lists
    sources = overlap_counts['naive_cluster'].map(naive_idx)
    targets = overlap_counts['hybrid_cluster'].map(hybrid_idx)
    values = overlap_counts['count']

    # Assign colors to nodes based on cl_Cat
    node_colors = [cl_Cat.get(str(l), '#cccccc') for l in labels]
    # Create Sankey plot
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=50,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        ))])

    # Add annotations for Naive LLM and Hybrid LLM with rotation
    # Use xref='paper', yref='paper' to keep annotations fixed relative to the plot area
    fig.add_annotation(
        x=0, y=1, xref='paper', yref='paper',
        text="Naive LLM clusters", showarrow=False, font=dict(size=14, color="black"),
        align='left'
    )
    fig.add_annotation(
        x=1, y=1, xref='paper', yref='paper',
        text="Hybrid LLM clusters", showarrow=False, font=dict(size=14, color="black"),
        align='right'
    )

    # fig.update_layout(title_text="Cluster Overlap: NaiveLLM vs Hybrid LLM (Sankey Plot)", font_size=12)
    fig.update_layout(
        # title_text="Cluster Overlap: NaiveLLM vs Hybrid LLM (Sankey Plot)",
        # font_size=12,
        width=300,   # set width in pixels
        height=400   # set height in pixels
    )
    # fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))  # Adjust margins to fit in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    # write plot     summary
    st.markdown('''
The Sankey plot above illustrates the overlap between Naive LLM clusters and Hybrid LLM clusters. Each node represents a cluster, 
and the width of the links indicates the number of genes that fall into both clusters. The colors of the nodes correspond to the 
clusters, performed using quantile clustering method of weighted gene scores from Naive LLM and Hybrid LLM evaluations.
This visualization helps in understanding how genes transition from one cluster to another during the evaluation process,
and it highlights the relationships between the clusters identified by the two methods.

This analysis is crucial for identifying genes that may have been misclassified or require further investigation,
as it provides insights into the consistency and reliability of the clustering results across different evaluation methods.

**weighted gene scores** : Each gene received scores (0-10 scale) across eight sepsis-related evaluation criteria, 
as mentioned (see manuscript). Individual criterion scores were categorized into three confidence bins: 
High (≥7), Medium (4-6), and Low (≤3). For each gene, the proportion of scores in each bin was calculated by dividing 
the count by the total number of criteria (n= 8). Weighted scores were computed using confidence-based weights: 
High × 1.0, Medium × 0.7, and Low × 0.3, with final scores ranging from 0.0 to 1.0. This weighted aggregation 
was essential to address the inherent unreliability of individual LLM scores by strategically rewarding genes that 
demonstrated consistent high-confidence evidence across multiple independent evaluation criteria. This approach helps in 
prioritizing robust multi-dimensional relevance over potentially misleading individual assessment. This methodology was 
applied consistently across naive LLM, RAG, and hybrid evaluation approaches to enable direct cross-method comparison.

''')
    ##############################################################################

    st.subheader('Candidate genes for Sepsis')
    st.markdown('''
- **PS1**: Genes that pass Naive LLM screening (score ≥5 in at least one criterion out of 8) [n= 609 out of >10K gene examined].
- **PS2**: Genes that are verified by RAG. We reprioritized these based on scoring and clustering. [n= 443 out of 609 PS1 genes].
- **PS3**: Top cluster of PS2 genes.[n=82 genes ]
- **Candidate genes**: Top cluster of PS3 genes across all criteria [n= 30 genes].
''')
    # Show candidate genes based on priority_cluster
    candidate_genes = stable[stable['priority_cluster'].notna()].index.tolist()
    # scatter plot for candidate genes
    if not candidate_genes:
        st.warning("No candidate genes found for scatter plot.")
    else:
        st.subheader('Scatter Plot of PS3 genes')
        # Prepare data for scatter plot
        candidate_df = stable[stable['priority_cluster'].notna()]
        priority_cluster_color = {1: '#ca0020', 2: '#f4a582',
                          3: '#b2df8a', 4: '#a6cee3',
                          5: '#1f78b4'}
        import plotly.express as px
        candidate_df = candidate_df.copy()
        candidate_df.index = candidate_df.index.astype(str)
        fig_scatter = px.scatter(
            candidate_df,
            x='PC1', y='PC2',
            color=candidate_df['priority_cluster'].astype(str),  # Make color discrete by casting to string
            hover_name=candidate_df.index,
            color_discrete_map={str(k): v for k, v in priority_cluster_color.items()},  # Ensure keys are strings
            title="PS3 Genes Scatter Plot",
            labels={'PC1': 'PC1', 'PC2': 'PC2', 'priority_cluster': 'Priority Cluster'}
        )
        fig_scatter.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        fig_scatter.update_layout(legend_title_text='Priority Cluster')
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        # plotly even is not work ing lets fix it later
        st.plotly_chart(fig_scatter, use_container_width=True)
    st.subheader('Gene list as per priority cluster')
    candidate_genes = stable[stable['priority_cluster']==1].index.tolist()
    if not candidate_genes:
        st.warning("No candidate genes found for gene list.")
    else:
        st.write(f"Total candidate genes in cluster 1 (n= {len(candidate_genes)})")
        # write gene_name in codeblock
        st.code('\n'.join(candidate_genes), language='plaintext')
    ps3_gene_list = stable[stable['HybridLLM_cluster']=="1"].index.tolist()
    if not ps3_gene_list:
        st.warning("No PS3 genes found for gene list.")
    else:
        st.write(f"Total PS3 genes (n= {len(ps3_gene_list)})")
        # Wrap the gene list to a fixed width (e.g., 80 chars per line)
        def wrap_gene_list(gene_list, width=80):
            s = ';'.join(gene_list)
            return '\n'.join([s[i:i+width] for i in range(0, len(s), width)])
        st.code(wrap_gene_list(ps3_gene_list), language='plaintext')
        
    ps2_gene_list = stable[stable['HybridLLM_cluster']!="Fail"].index.tolist()
    if not ps2_gene_list:
        st.warning("No PS2 genes found for gene list.")
    else:
        st.write(f"Total PS2 genes (n= {len(ps2_gene_list)})")
        st.code(wrap_gene_list(ps2_gene_list), language='plaintext')



elif page == 'Gene-wise Detail':
    st.title('Gene-wise Detail Report')
    st.sidebar.header('Filter')
    selected_gene = st.sidebar.selectbox('Select Gene Name', gene_names)
    # Filter dataframe based on selected gene
    df_filtered = df[df[gene_col] == selected_gene].reset_index(drop=True)
    if df_filtered.empty:
        st.write(f'No data found for gene: {selected_gene}')
        st.stop()
    # Show filtered dataframe
    st.subheader(f'Data for Gene: {selected_gene}')
    df_to_Show = df_filtered[[gene_col, json_key_col, 'RAG_Evaluation_Result', 'naive_score', 'final_score', 'scientific_explanation']]
    # color mapping for RAG_Evaluation_Result
    status_colors = {'Pass': '#1b9e77', 'Fail': '#d95f02'}
    def color_status(val):
        if val == 'Pass':
            return 'background-color: #1b9e77; color: white'
        elif val == 'Fail':
            return 'background-color: #d95f02; color: white'
        else:
            return ''
    # Wrap scientific_explanation column to fixed width using HTML
    def wrap_scientific(val):
        return f'<div style="white-space:pre-wrap;word-break:break-word;max-width:1500px">{val}</div>'
    styled_df = df_to_Show.copy()
    # styled_df['scientific_explanation'] = styled_df['scientific_explanation'].astype(str).apply(wrap_scientific)
    st.write(
        styled_df.style
            .map(color_status, subset=['RAG_Evaluation_Result'])
            .to_html(escape=False),
        unsafe_allow_html=True
    )

    # Show summary statistics for the selected gene
    selected_json_key = st.sidebar.selectbox('Select JSON Key', json_keys)
    # Default columns for detailed view
    all_cols = list(df.columns)
    default_detail_cols = [
        'final_score', 'scientific_explanation', 'naive_score',
        'naive_justification', 'rag_score', 'rag_justification', 'reference'
    ]
    default_detail_cols = [col for col in default_detail_cols if col in all_cols]
    # Column selection for detailed view
    st.subheader('Detailed Row View')
    detail_cols = st.multiselect('Select columns to display:', all_cols, default=default_detail_cols)
    row = df[(df[gene_col] == selected_gene) & (df[json_key_col] == selected_json_key)]
    if not row.empty:
        transposed = row[[gene_col, json_key_col] + detail_cols].T
        transposed.columns = ['Value']
        transposed.index.name = 'Field'
        def wrap_text(val):
            return f'<div style="white-space:pre-wrap;word-break:break-word;max-width:1200px">{val}</div>'
        transposed_display = transposed.copy()
        transposed_display['Value'] = transposed_display['Value'].astype(str).apply(wrap_text)
        st.markdown(
            transposed_display.to_html(escape=False),
            unsafe_allow_html=True
        )
    else:
        st.write('No data found for this gene and key.')
    # rag_noderetrived details
    if 'rag_noderetrived' in df.columns:
        st.subheader('rag_noderetrived Details for Selected Row')
        if not row.empty:
            rag_nodes = row.iloc[0]['rag_noderetrived']
            if isinstance(rag_nodes, list):
                for i, k in enumerate(rag_nodes):
                    node_id = getattr(k, 'node_id', '')
                    text = getattr(k, 'text', '')
                    meta = getattr(k, 'metadata', {})
                    ref = meta['Reference'] if isinstance(meta, dict) and 'Reference' in meta else ''
                    st.markdown(
                        f"<span style='font-family:monospace; font-size:0.85em;'><b>Node ID:</b> {node_id}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='font-family:monospace; font-size:0.85em; white-space:pre-wrap; word-break:break-word; max-width:1400px'><b>Text:</b> {text}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<span style='font-family:monospace; font-size:0.85em;'><b>Reference:</b> {ref}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown('---')
            else:
                st.write('rag_noderetrived is not a list')
        else:
            st.write('No data found for this gene and key.')

elif page == 'Resources':
    st.title('Resources')
    st.markdown('''
    ## Get the Index vector database from Zenodo
    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15802241.svg)](https://doi.org/10.5281/zenodo.15802241)

    ## Set up query using notebook
    - You can use the notebook `3_runPipeLine.ipynb` to set up the query for the RAG evaluation pipeline. This notebook provides a step-by-step guide on how to
    configure and run the RAG evaluation pipeline using the pre-built vector database.

    ## see repo for more details
    - For more details on the methodology and implementation, please refer to the [GitHub repository](https://github.com/taushifkhan/llm-geneprioritization-framework).
    - The repository contains the code, data, and documentation for the RAG evaluation pipeline and the candidate gene selection workflow.
    ## Contact
    - For any questions or issues, please contact the authors via the GitHub repository or email us at [taushifkhan](mailto:taushifkhan@jax.org).
    ''')

