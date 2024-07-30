import os
import gradio as gr
from gradio.components import Textbox, Button, Slider, Checkbox
from AinaTheme import theme
from urllib.error import HTTPError

from rag import RAG
from utils import setup

MAX_NEW_TOKENS = 700
SHOW_MODEL_PARAMETERS_IN_UI = os.environ.get("SHOW_MODEL_PARAMETERS_IN_UI", default="True") == "True"

setup()


rag = RAG(
    hf_token=os.getenv("HF_TOKEN"),
    embeddings_model=os.getenv("EMBEDDINGS"), 
    model_name=os.getenv("MODEL"),   
    

)


def generate(prompt, model_parameters):
    try:
        output, context, source = rag.get_response(prompt, model_parameters)
        return output, context, source
    except HTTPError as err:
        if err.code == 400:
            gr.Warning(
                "The inference endpoint is only available Monday through Friday, from 08:00 to 20:00 CET."
            )
    except:
        gr.Warning(
            "Inference endpoint is not available right now. Please try again later."
        )


def submit_input(input_, num_chunks, max_new_tokens, repetition_penalty, top_k, top_p, do_sample, temperature):
    if input_.strip() == "":
        gr.Warning("Not possible to inference an empty input")
        return None


    model_parameters = {
        "NUM_CHUNKS": num_chunks,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "top_k": top_k,
        "top_p": top_p,
        "do_sample": do_sample,
        "temperature": temperature
    }

    output, context, source = generate(input_, model_parameters)
    sources_markup = ""

    for url in source:
        sources_markup += f'<a href="{url}" target="_blank">{url}</a><br>'
        
    return output.strip(), sources_markup, context


def change_interactive(text):
    if len(text) == 0:
        return gr.update(interactive=True), gr.update(interactive=False)
    return gr.update(interactive=True), gr.update(interactive=True)


def clear():
    return (
        None, 
        None,
        None,
        None,
        gr.Slider(value=2.0),
        gr.Slider(value=MAX_NEW_TOKENS),
        gr.Slider(value=1.0),
        gr.Slider(value=50),
        gr.Slider(value=0.99),
        gr.Checkbox(value=False),
        gr.Slider(value=0.35),
    )


def gradio_app():
    with gr.Blocks(theme=theme) as demo:
        with gr.Row():
            with gr.Column(scale=0.1):
                gr.Image("rag_image.jpg", elem_id="flor-banner", scale=1, height=256, width=256, show_label=False, show_download_button = False, show_share_button = False)
            with gr.Column():
                gr.Markdown(
                    """# Demo de Retrieval-Augmented Generation per documents legals
                    🔍 **Retrieval-Augmented Generation** (RAG) és una tecnologia de IA que permet interrogar un repositori de documents amb preguntes 
                    en llenguatge natural, i combina tècniques de recuperació d'informació avançades amb models generatius per redactar una resposta 
                    fent servir només la informació existent en els documents del repositori. 
                        
                    🎯 **Objectiu:** Aquest és un primer demostrador amb la normativa vigent publicada al Diari Oficial de la Generalitat de Catalunya, en el 
                    repositori del EADOP (Entitat Autònoma del Diari Oficial i de Publicacions). Aquesta primera versió explora prop de 2000 documents en català, 
                    i genera la resposta fent servir el model Flor6.3b entrenat amb el dataset de QA generativa projecte-aina/RAG_Multilingual. 
                    
                    ⚠️ **Advertencies**: Primera versió experimental. El contingut generat per aquest model no està supervisat i pot ser incorrecte. 
                    Si us plau, tingueu-ho en compte quan exploreu aquest recurs.                 
                    """
                )
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                input_ = Textbox(
                    lines=11,
                    label="Input",
                    placeholder="Quina és la finalitat del Servei Meteorològic de Catalunya?",
                    # value = "Quina és la finalitat del Servei Meteorològic de Catalunya?"
                )
                with gr.Row(variant="panel"):
                    clear_btn = Button(
                        "Clear",
                    )
                    submit_btn = Button("Submit", variant="primary", interactive=False)

                with gr.Row(variant="panel"):
                    with gr.Accordion("Model parameters", open=False, visible=SHOW_MODEL_PARAMETERS_IN_UI):
                        num_chunks = Slider(
                            minimum=1,
                            maximum=6,
                            step=1,
                            value=2,
                            label="Number of chunks"
                        )
                        max_new_tokens = Slider(
                            minimum=50,
                            maximum=2000,
                            step=1,
                            value=MAX_NEW_TOKENS,
                            label="Max tokens"
                        )
                        repetition_penalty = Slider(
                            minimum=0.1,
                            maximum=2.0,
                            step=0.1,
                            value=1.0,
                            label="Repetition penalty"
                        )
                        top_k = Slider(
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=50,
                            label="Top k"
                        )
                        top_p = Slider(
                            minimum=0.01,
                            maximum=0.99,
                            value=0.99,
                            label="Top p"
                        )  
                        do_sample = Checkbox(
                            value=False, 
                            label="Do sample"
                        )
                        temperature = Slider(
                            minimum=0.1, 
                            maximum=1,
                            value=0.35,
                            label="Temperature"
                        )

                        parameters_compontents = [num_chunks, max_new_tokens, repetition_penalty, top_k, top_p, do_sample, temperature]

            with gr.Column(variant="panel"):
                output = Textbox(
                    lines=10, 
                    label="Output", 
                    interactive=False, 
                    show_copy_button=True
                )
                with gr.Accordion("Sources and context:", open=False):
                    source_context = gr.Markdown(
                        label="Sources",
                        show_label=False,
                    )
                    with gr.Accordion("See full context evaluation:", open=False):
                        context_evaluation = gr.Markdown(
                            label="Full context",
                            show_label=False,
                            # interactive=False, 
                            # autoscroll=False,
                            # show_copy_button=True
                        )
                

        input_.change(
            fn=change_interactive,
            inputs=[input_],
            outputs=[clear_btn, submit_btn],
            api_name=False,
        )

        input_.change(
            fn=None,
            inputs=[input_],
            api_name=False,
            js="""(i, m) => {
            document.getElementById('inputlenght').textContent = i.length + '  '
            document.getElementById('inputlenght').style.color =  (i.length > m) ? "#ef4444" : "";
        }""",
        )

        clear_btn.click(
            fn=clear, 
            inputs=[], 
            outputs=[input_, output, source_context, context_evaluation] + parameters_compontents,
              queue=False, 
              api_name=False
        )
        
        submit_btn.click(
            fn=submit_input, 
            inputs=[input_]+ parameters_compontents, 
            outputs=[output, source_context, context_evaluation],
            api_name="get-results"
        )

        with gr.Row():
            with gr.Column(scale=0.5):
                gr.Examples(
                    examples=[
                        ["""Què és l'EADOP (Entitat Autònoma del Diari Oficial i de Publicacions)?"""],
                    ],
                    inputs=input_,
                    outputs=[output, source_context, context_evaluation],
                    fn=submit_input,
                )
                gr.Examples(
                    examples=[
                        ["""Què diu el decret sobre la senyalització de les begudes alcohòliques i el tabac a Catalunya?"""],
                    ],
                    inputs=input_,
                    outputs=[output, source_context, context_evaluation],
                    fn=submit_input,
                )
                gr.Examples(
                    examples=[
                        ["""Com es pot inscriure una persona al Registre de catalans i catalanes residents a l'exterior?"""],
                    ],
                    inputs=input_,
                    outputs=[output, source_context, context_evaluation],
                    fn=submit_input,
                )
                gr.Examples(
                    examples=[
                        ["""Quina és la finalitat del Servei Meterològic de Catalunya ?"""],
                    ],
                    inputs=input_,
                    outputs=[output, source_context, context_evaluation],
                    fn=submit_input,
                )

        demo.launch(show_api=True)


if __name__ == "__main__":
    gradio_app()