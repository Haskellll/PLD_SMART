import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time

st.sidebar.title('Options')
options = st.sidebar.radio('', ['All jobs', 'Select jobs'], 0)

if options == 'All jobs':
    st.title('All jobs')
    st.write('You accept to compute all incoming jobs')
elif options == 'Select jobs':
    st.title('Select jobs')
    st.write('You can select the jobs you want to compute')

    table = pd.DataFrame({
        'Job description': ['Description'],
        'Size': [10]
    })
    st.write(table)

    if st.button('Accept'):
        st.write('Accepted !')
    