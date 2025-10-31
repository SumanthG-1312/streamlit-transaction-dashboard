import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Transaction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("💰 Transaction Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('transactions.csv')
        # Convert date column to datetime if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()

if df is not None:
    # Sidebar filters
    st.sidebar.header("📊 Filters")
    
    # Display basic info about the dataset
    st.sidebar.info(f"**Total Transactions:** {len(df):,}")
    
    # Show column names for debugging/reference
    with st.sidebar.expander("📋 Dataset Info"):
        st.write("**Columns:**", list(df.columns))
        st.write("**Shape:**", df.shape)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💹 Analytics", "📊 Visualizations", "📋 Data Table"])
    
    with tab1:
        st.header("Transaction Overview")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Transactions",
                value=f"{len(df):,}",
                delta=None
            )
        
        # Try to find amount/value column
        amount_col = None
        for col in ['amount', 'Amount', 'value', 'Value', 'transaction_amount', 'price', 'Price']:
            if col in df.columns:
                amount_col = col
                break
        
        if amount_col:
            with col2:
                total_amount = df[amount_col].sum()
                st.metric(
                    label="Total Amount",
                    value=f"${total_amount:,.2f}",
                    delta=None
                )
            
            with col3:
                avg_amount = df[amount_col].mean()
                st.metric(
                    label="Average Amount",
                    value=f"${avg_amount:,.2f}",
                    delta=None
                )
            
            with col4:
                max_amount = df[amount_col].max()
                st.metric(
                    label="Largest Transaction",
                    value=f"${max_amount:,.2f}",
                    delta=None
                )
        
        # Display first few rows
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.header("Analytics & Insights")
        
        if amount_col:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Transaction Amount Distribution")
                fig_hist = px.histogram(
                    df, 
                    x=amount_col, 
                    nbins=30,
                    title="Distribution of Transaction Amounts"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.subheader("Transaction Amount Statistics")
                stats = df[amount_col].describe()
                st.dataframe(stats.to_frame(name='Statistics'), use_container_width=True)
        
        # Try to find categorical columns for analysis
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            st.subheader("Categorical Analysis")
            selected_cat = st.selectbox("Select a categorical column to analyze:", categorical_cols)
            
            if selected_cat:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Value counts
                    value_counts = df[selected_cat].value_counts().head(10)
                    st.write(f"**Top 10 {selected_cat} values:**")
                    st.dataframe(value_counts.to_frame(name='Count'), use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig_bar = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Distribution of {selected_cat}"
                    )
                    fig_bar.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.header("Data Visualizations")
        
        # Let users select columns for visualization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("Select X-axis:", numeric_cols)
            
            with col2:
                y_axis = st.selectbox("Select Y-axis:", [col for col in numeric_cols if col != x_axis])
            
            if x_axis and y_axis:
                # Scatter plot
                fig_scatter = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    title=f"{x_axis} vs {y_axis}",
                    opacity=0.7
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Time series plot if date column exists
        date_cols = [col for col in df.columns if 'date' in col.lower() or df[col].dtype == 'datetime64[ns]']
        if date_cols and amount_col:
            st.subheader("Time Series Analysis")
            date_col = date_cols[0]
            
            # Group by date and sum amounts
            daily_amounts = df.groupby(df[date_col].dt.date)[amount_col].sum().reset_index()
            daily_amounts.columns = ['Date', 'Total_Amount']
            
            fig_line = px.line(
                daily_amounts,
                x='Date',
                y='Total_Amount',
                title="Transaction Amounts Over Time"
            )
            st.plotly_chart(fig_line, use_container_width=True)
    
    with tab4:
        st.header("Complete Data Table")
        
        # Add search functionality
        search_term = st.text_input("🔍 Search in data:", "")
        
        if search_term:
            # Filter data based on search term
            mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            filtered_df = df[mask]
            st.write(f"Found {len(filtered_df)} matching records:")
        else:
            filtered_df = df
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            rows_to_show = st.selectbox("Rows to display:", [10, 25, 50, 100, "All"], index=1)
        
        with col2:
            columns_to_show = st.multiselect(
                "Select columns to display:",
                options=list(df.columns),
                default=list(df.columns)
            )
        
        if columns_to_show:
            display_df = filtered_df[columns_to_show]
            
            if rows_to_show != "All":
                display_df = display_df.head(rows_to_show)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download filtered data as CSV",
                data=csv,
                file_name=f"filtered_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
else:
    st.error("❌ Could not load transactions.csv file. Please make sure the file exists in the same directory as this app.")
    st.info("💡 The app expects a CSV file named 'transactions.csv' in the same directory.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        📊 Transaction Dashboard | Built with Streamlit ❤️
    </div>
    """,
    unsafe_allow_html=True
)
