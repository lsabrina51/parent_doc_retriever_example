# Parent Document Retriever Examples
Overview: Demonstrates Langchain's parent documentation retriever wuth streamlit frontends.

## Contents
Set-up: Download dependent libraries by ```pip install -r requirement.txt```

There are 4 programs: 

## Engineering Textbook
Uses engin.pdf as source material. Has two programs. 

1. Engineering Normal (engin_norm)- uses normal retriever
    - Run with ```python engin_norm_streamlit.py```
3. Engineering PDR (engin_pdr)- uses the parent documentation retriever
    - Run with ```python engin_pdr_streamlit.py```

## UOfM Example
Uses the umich-example.pdf as materials

1. Umich Normal (umich_norm)- uses normal retriever
    - Run with ```python umich_norm_streamlit.py```
3. Umich PDR (umich_pdr)- uses the parent documentation retriever
    - Run with ```python umich_pdr_streamlit.py```
  
##Notes
PDR programs tend to be more contextualized and has more facts even when the query is specific. 
