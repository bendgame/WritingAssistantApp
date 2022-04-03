#import dependencies
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForCausalLM
from readability import Readability
import nltk

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained("gpt2")

gen_text_list = []
exv = 0

#create an input field
def textareas():
    return html.Div([ 
            dbc.Textarea(id = 'my-input'
                , size="lg"
                , placeholder="Enter text for auto completion")
            , dbc.Button("Submit"
                , id="gen-button"
                , className="me-2"
                , n_clicks=0)
            ])


#instantiate dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#create layout
app.layout = html.Div([dbc.Container([
        html.H1("Eric's Writing Assistant")
        , html.Br()
        , html.H3("Enter a prompt")
        , textareas()
        , html.Br()
        , html.Br()
        , html.H3("Generated Text")
        , html.Div(id='readability-score')
        , html.Div(id='my-output')
        , dbc.Button("Expand", id="expand-button", className="me-2", n_clicks=0)
        , dbc.Button("Clear", id="clear-button", className="me-2", n_clicks=0)
        
   ])
  ])

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Output(component_id = 'readability-score', component_property='children'),
    Input(component_id='gen-button', component_property='n_clicks'),
    Input(component_id='expand-button', component_property='n_clicks'),
    Input(component_id='clear-button', component_property='n_clicks'),
    State(component_id='my-input', component_property='value')    
)
def update_output_div(gen, ex, cl, input_value):
    gen_text = ""
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    global gen_text_list
    global exv
    score = ''
    if 'gen-button' in changed_id:
        
        if input_value is None or input_value == "":
            input_value = ""
            gen_text = ""

        else:
            
            input_ids = tokenizer(input_value, return_tensors="pt").input_ids

            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                max_length=100,
            )
            gen_text = tokenizer.batch_decode(gen_tokens)[0]
            
            gen_text_list.append(gen_text)

            if len(gen_text.strip().split(" ")) >100:
                print(len(gen_text))
                r = Readability(gen_text)
                fk = r.flesch_kincaid()
                score = fk.score
                
            else: 
                score = 'Not 100 tokens'
    
    if 'expand-button' in changed_id:

        if len(gen_text_list) > 0:
            MAX_LENGTH = 100 + 100*(exv+1)
            input_ids = tokenizer(gen_text_list[exv], return_tensors="pt").input_ids

            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                max_length=MAX_LENGTH,
            )
            gen_text = tokenizer.batch_decode(gen_tokens)[0]
           
            gen_text_list.append(gen_text)
            exv+=1
        
            if len(gen_text.strip().split(" ")) >100:
                print(len(gen_text))
                r = Readability(gen_text)
                fk = r.flesch_kincaid()
                score = fk.score
                
            else: 
                score = 'Not 100 tokens'


        else:
            html.P("no text has been generated")

    if 'clear-button' in changed_id:
        gen_text = ''
        exv = 0
        gen_text_list = []
    
    return html.P(gen_text), html.P(f"Readability Score: {score}")


    

#run app server
if __name__ == '__main__':
    app.run_server(debug=True)
