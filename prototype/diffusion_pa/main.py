from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
import torch 
import time
import psutil
import os
import functools

# conda create -n "pelugens" python=3.11 -y && conda activate pelugens
# pip3 install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
# pip install transformers diffusers accelerate



# IGNORAR ESSE BLOCO ============ VAI PRA BAIXO

def resource_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        cpu_before = process.cpu_percent(interval=None)
        memory_before = process.memory_info().rss / 1024 ** 2 

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        cpu_after = process.cpu_percent(interval=None)
        memory_after = process.memory_info().rss / 1024 ** 2 
        
        print(f"Exec Time: {end_time - start_time:.4f} seconds")
        print(f"CPU Usage: {cpu_after - cpu_before:.2f}%")
        print(f"Memory Usage: {memory_after - memory_before:.2f} MB")
        
        return result

    return wrapper

# FIM DA IGNORÂNCIA ============ VAI PRA BAIXO

# ============ COMECE AQUI ============




# ============ DECLARAÇÃO DE MODELO ============
"""
    Aqui estamos usando uma Pipeline da HuggingFace pra Text2Image.
    Nem todos os nomes na hugging são tão descritivos. 
    Provavelmente o que estamos é um "StableDiffusionPipeline" de Image2Image.
    Se você encontrar um que tenha Image+Text2Image, ta tudo bem, por que podemos
    settar o hiperparametro "guidance_scale" pra 1.0 (basicamente nenhuma influência).
    
    Se você tiver bastante VRAM, pode usar uma precisão maior que f16, vai fundo.

    ps: não confie no GPT pra te sugerir modelos.
"""
model = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
model.to("cuda")
model.safety_checker = None


# ============ LAYER HOOKS ============
def backward_hook_lookup(module, grad_input, grad_output):
    """
    Esse é um hook pra ver o backfire da AI durante a inferencia.
    Aqui temos acesso aos gradientes. Não acho que seja o que precisamos agora.
    """
    print(f"Grad Input: {grad_input}")
    print(f"Grad Output: {grad_output}")


def forward_hook_lookup(module, input, output):
    """
    Esse é um hook pro forward do modelo durante inferência.
    Aqui temos acesso aos pesos (input e output).
    Lembrando que o input de uma layer é o output da outra.
    
    ps: Esses prints são gigantes, eu to printando só o inicio
    de cada um e já toma o console todo.
    """
    # prints the layer name origin
    print(f"\n\nLayer: {module}")
    print(f"""
        ======================================
        Input Shape: {input[0].shape},
        Output Shape: {output.shape},
        =======================================
    """)

    # adicione esses se quiser printar as layers,
    # tirei pra não ficar bagunçado os prints
        # Input: [{input[2:][1:]} ... {input[:2][:1]}],
        # Output: [{output[2:][1:]} ... {output[:2][:1]}]


# ============ USANDO OS HOOKS ============
"""
Seguinte mano, essa linha abaixo aqui é a extração de uma layer específica do modelo (a primeira no caso).
A ideia é pegarmos algumas layers e registrar os pesos delas.

Primeiro precisamos contar quantas layers possuímos, vamos supor 128.
Como discutimos no quadro, seria bom dividir elas em algo como fator 8.
128/8 = 16
[0 ... 64 ... 128]
Então por exemplo, poderíamos pegar 1 layer a cada 8, terminando com 16 (inclusivo).
Ou dividir 16 por 3 por exemplo, pegando 5 layers do começo, 5 do meio e 5 do final.
Pode explorar esses números, up to you.
"""
layer = model.unet.down_blocks[0].resnets[0] 

# guardando as layers e handles
layers:list = []
handles:list = []

def register_hook(layer, callback=forward_hook_lookup):
    """
    Aqui a gente ta criando um handle pra injetar no registro da layer
    um hook. 

    a.k.a A gente ta falando que toda vez que a layer chamar a função 
    forward(), ela vai também chamar a função "forward_hook_lookup" 
    e se injetar como parametros nessa função (no caso, "module, input, output").
    """
    handle = layer.register_forward_hook(callback)
    return handle

layers.append(layer)
handles.append(register_hook(layers[0]))



# ============ INFERENCIA ============
# inference example
@resource_usage
def inferir(prompt, model=model, output_name="out_img.png"):
    output = model(prompt)
    output.images[0].save(output_name)

inferir("a maior piranha de todos os tempos (não o peixe, a quenga no caso) 4k high quality photorealistic")

def remove_handles(handles:list=handles):
    """
    Vamos remover os handles aqui pra impedir vazamento de memória.
    """
    for handle in handles:
        handle.remove()
remove_handles()


# ============ O QUE QUEREMOS & TEORIA ============
"""

Qual o objetivo?

Nós queremos salvar os outputs de uma quantidade X de layers, para podermos usar
sua extração de features que acontece naturalmente durante a inferência.
Isso é comum em modelos convolutivos, mas queremos fazer isso com um modelo difusor.

---

Por que não salvamos todas as Layers?

Pois não teremos memória infinita na cloud, e nem tempo infinito pra treinar o modelo
classificador depois.

---

Como sabemos quantas layers salvar?

Fazendo a seguinte conta:

Quando printamos uma UNICA layer, vamos obter o seguinte
    Input Shape: torch.Size([2, 320, 64, 64]),
    Output Shape: torch.Size([2, 320, 64, 64]),

Significa que temos uma array com
2 * 320 * 64 * 64 = 2.621.440 elementos
Esses elementos tem uma precisão de 16bits flutuantes, vulgo 2 bytes por elemento.
Então se multiplicarmos os dois:
2.621.440 * 2 = 5.242.880 bytes

Convertendo pra megabytes:
5.242.880 / (1024 * 1024) = 5mb

Agora multiplicamos isso pra quantidade de layer que queremos, vamos supor 16.
5mb * 16L = 80mb

Por final, vamos multiplicar isso pelo tamanho do nosso dataset, vamos supor 500 imagens:
80mb * 500img = 40.000mb

Pois multipliquemos isso por 1/1024 novamente para termos isso em Gigabytes.
40.000 * (1/1024) = 39,0625 GB

Sim, salvando 16 layers, nosso dataset possui 39 gigabytes. '-'

O ideal seria nós salvarmos eles comprimidos, vc pode ver um exemplo em
`compressão_salvar_tensor_exemplo.py`

Não vai lá ajudar de muita coisa, mas já ajuda indo pra mais ou menos 34GB.

---


"""

