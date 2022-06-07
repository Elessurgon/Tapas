import tensorflow.compat.v1 as tf
import os
import shutil
import csv
import sys
import pandas as pd
import numpy as np
import IPython
import streamlit as st
# from st_aggrid import AgGrid
#import subprocess
from itertools import islice
import random
# from transformers import pipeline
from transformers import TapasTokenizer, TapasForQuestionAnswering
# from transformers import TapexTokenizer, BartForConditionalGeneration

tf.get_logger().setLevel('ERROR')


model_name = 'google/tapas-large-finetuned-wtq'
# model_name = 'microsoft/tapex-large-finetuned-wikisql'
#model_name =  "table-question-answering"
#model = pipeline(model_name)

model = TapasForQuestionAnswering.from_pretrained(
    model_name, local_files_only=False)
tokenizer = TapasTokenizer.from_pretrained(model_name)

# model = BartForConditionalGeneration.from_pretrained(
#     model_name, local_files_only=False)
# tokenizer = TapexTokenizer.from_pretrained(model_name)

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Query your Table')
st.header('Upload CSV file')

uploaded_file = st.file_uploader("Choose your CSV file", type='csv')
placeholder = st.empty()

data = None

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.replace(',', '', regex=True, inplace=True)
    if st.checkbox('Want to see the data?'):
        placeholder.dataframe(data)

st.header('Enter your queries')
input_queries = st.text_input(
    'Type your queries separated by comma(,)', value='')
input_queries = input_queries.split(',')

colors1 = ["#"+''.join([random.choice('0123456789ABCDEF')
                       for j in range(6)]) for i in range(len(input_queries))]
colors2 = ['background-color:' +
           str(color)+'; color: black' for color in colors1]


def styling_specific_cell(x, tags, colors):
    df_styler = pd.DataFrame('', index=x.index, columns=x.columns)
    for idx, tag in enumerate(tags):
        for r, c in tag:
            df_styler.iloc[r, c] = colors[idx]
    return df_styler


if st.button('Predict Answers'):
    with st.spinner('It will take approx a minute'):
        table = data.astype(str)
        inputs = tokenizer(table=table, queries=input_queries,
                           padding='max_length', truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        #outputs = model(table = data, query = queries)
        predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
            inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())

        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        aggregation_predictions_string = [
            id2aggregation[x] for x in predicted_aggregation_indices]

        answers = []

        for coordinates in predicted_answer_coordinates:
            if len(coordinates) == 1:
                # only a single cell:
                answers.append(table.iat[coordinates[0]])
            else:
                # multiple cells
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(table.iat[coordinate])
                answers.append(", ".join(cell_values))

try:
    st.show(
        'Done! Please check below the answers and its cells highlighted in table above')

    placeholder.dataframe(data.style.apply(
        styling_specific_cell, tags=predicted_answer_coordinates, colors=colors2, axis=None))

    for query, answer, predicted_agg, c in zip(input_queries, answers, aggregation_predictions_string, colors1):
        st.write('\n')
        st.markdown('<font color={} size=4>**{}**</font>'.format(c,
                    query), unsafe_allow_html=True)
        st.write('\n')

        if predicted_agg == "NONE" or predicted_agg == 'COUNT':
            st.markdown('**>** '+str(answer))
        else:
            # st.write(predicted_agg)
            # st.write(answer)
            if predicted_agg == 'SUM':
                st.markdown(
                    '**>** '+str(sum(list(map(float, answer.split(','))))))
            else:
                st.markdown(
                    '**>** '+str(np.round(np.mean(list(map(float, answer.split(',')))), 2)))
except:
    st.warning('None')
