
# Entendendo um Transformer do Zero: Um Guia de Código

Este documento explica a arquitetura de um modelo Transformer que foi implementada em Python usando PyTorch. O objetivo é servir como um guia de aprendizado, detalhando o propósito de cada arquivo e como eles se conectam para formar um Modelo de Linguagem Grande (LLM) capaz de responder a perguntas.

A arquitetura é baseada no paper seminal "Attention Is All You Need".

## Visão Geral dos Arquivos

O projeto está dividido em cinco arquivos principais:

1.  **`encoder.py`**: Define a arquitetura do **Codificador**. Sua função é "entender" a sequência de entrada (a pergunta).
2.  **`decoder.py`**: Define a arquitetura do **Decodificador**. Sua função é gerar a sequência de saída (a resposta), uma palavra de cada vez.
3.  **`transformer.py`**: Une o Codificador e o Decodificador para formar o modelo Transformer completo.
4.  **`train.py`**: Contém o código para treinar nosso modelo Transformer usando um dataset de perguntas e respostas.
5.  **`predict.py`**: Contém o código para carregar o modelo treinado e usá-lo para fazer previsões (inferência) em novas perguntas.

---

## 1. `encoder.py` - O Codificador

**Propósito:** A principal tarefa do Codificador é processar a sequência de entrada (uma pergunta) e transformá-la em uma representação numérica rica em contexto. Ele não apenas entende o significado de cada palavra, mas também como as palavras se relacionam umas com as outras na frase.

### Componentes Chave:

*   **`PositionalEncoding` (Codificação Posicional):** Transformers, por natureza, não entendem a ordem das palavras. Esta classe injeta informações sobre a posição de cada palavra na sequência usando funções de seno e cosseno. Isso é crucial para que o modelo saiba que "o gato persegue o rato" é diferente de "o rato persegue o gato".

*   **`MultiHeadAttention` (Atenção Multi-Cabeças):** Este é o coração do Transformer.
    *   **Auto-Atenção (Self-Attention):** Permite que cada palavra na frase "olhe" para todas as outras palavras na *mesma* frase. Isso ajuda o modelo a entender o contexto. Por exemplo, na frase "Eu fui ao banco sacar dinheiro", a atenção ajuda o modelo a entender que "banco" se refere a uma instituição financeira, e não a um assento.
    *   **Múltiplas Cabeças:** A "atenção" é realizada várias vezes em paralelo (múltiplas "cabeças"). Cada cabeça pode aprender a focar em diferentes tipos de relações entre as palavras (ex: uma cabeça pode focar em relações gramaticais, outra em relações semânticas).

*   **`PositionwiseFeedForward` (Rede Feed-Forward por Posição):** Uma rede neural simples que é aplicada a cada palavra (posição) de forma independente. Ela processa a informação rica em contexto que foi coletada pela camada de atenção.

*   **`EncoderLayer` e `Encoder`:** Um `EncoderLayer` simplesmente agrupa uma camada de `MultiHeadAttention` e uma `PositionwiseFeedForward`. O `Encoder` final é uma pilha de vários desses `EncoderLayer`s. Quanto mais camadas, mais "profundo" é o entendimento do modelo sobre a linguagem.

---

## 2. `decoder.py` - O Decodificador

**Propósito:** A tarefa do Decodificador é gerar a sequência de saída (a resposta) palavra por palavra. Ele faz isso usando a representação da pergunta (criada pelo Codificador) e as palavras que ele já gerou na resposta.

### Componentes Chave:

O Decodificador é semelhante ao Codificador, mas com uma diferença crucial em suas camadas de atenção:

1.  **`Masked Multi-Head Attention` (Atenção Multi-Cabeças Mascarada):** Esta é a camada de auto-atenção do Decodificador. Ela permite que cada palavra na resposta "olhe" para as palavras *anteriores* na mesma resposta. A "máscara" é a parte importante: ela impede que o modelo "trapaceie" olhando para palavras futuras na resposta que ele ainda não deveria ter gerado.

2.  **`Encoder-Decoder Attention` (Atenção Codificador-Decodificador):** É aqui que a mágica acontece. Esta camada permite que o Decodificador olhe para a saída do Codificador (a representação da pergunta). A cada passo, o Decodificador usa a informação da pergunta para decidir qual é a melhor palavra a ser gerada a seguir na resposta. É assim que a resposta é condicionada à pergunta.

---

## 3. `transformer.py` - O Modelo Completo

**Propósito:** Este arquivo é a "cola" que une tudo.

### Componentes Chave:

*   **`Encoder`**: Uma instância do Codificador de `encoder.py`.
*   **`Decoder`**: Uma instância do Decodificador de `decoder.py`.
*   **`nn.Linear` (Camada de Saída):** A saída do Decodificador é um vetor de números para cada palavra. Esta camada linear final atua como um classificador. Ela transforma esse vetor em uma distribuição de probabilidade sobre todo o vocabulário de palavras possíveis. A palavra com a maior probabilidade é a previsão do modelo para o próximo passo.

---

## 4. `train.py` - O Treinamento

**Propósito:** Ensinar nosso modelo a responder perguntas, mostrando-lhe exemplos.

### Processo:

1.  **Carregar e Pré-processar Dados:** O script lê os arquivos `questions.txt` e `answers.txt`. Ele limpa o texto e cria dois **vocabulários**: um para as perguntas e outro para as respostas. Um vocabulário é simplesmente um mapa que converte cada palavra única em um número (token).

2.  **`Dataset` e `DataLoader`:** O PyTorch usa essas classes para gerenciar os dados. O `DataLoader` agrupa os dados em lotes (batches) e os embaralha, o que é essencial para um treinamento eficiente.

3.  **O Loop de Treinamento:** O script itera sobre o dataset várias vezes (épocas). Em cada passo:
    a.  Pega um lote de perguntas e respostas.
    b.  Passa os dados pelo modelo para obter uma previsão.
    c.  Calcula o **erro (Loss)** entre a previsão do modelo e a resposta real.
    d.  Usa o **otimizador (Adam)** para ajustar os pesos internos do modelo, de modo a reduzir o erro na próxima vez.
    e.  Este processo é repetido até que o erro seja minimizado.

4.  **Salvando o Modelo:** Após o treinamento, os pesos aprendidos pelo modelo são salvos no arquivo `transformer_qa.pth`.

---

## 5. `predict.py` - A Inferência

**Propósito:** Usar o modelo treinado para responder a novas perguntas.

### Processo:

1.  **Carregar o Modelo e os Vocabulários:** O script primeiro carrega o arquivo `transformer_qa.pth` e recria os vocabulários a partir dos dados de treinamento.

2.  **Processo de Geração (Auto-Regressivo):** Este é o núcleo da inferência:
    a.  A pergunta do usuário é processada e passada pelo **Codificador** *uma única vez*.
    b.  O **Decodificador** começa com um token de início de sentença (`<SOS>`).
    c.  Entra em um loop:
        i.  O modelo prevê a próxima palavra da resposta com base na saída do Codificador e nas palavras que já gerou.
        ii. A palavra prevista é adicionada à entrada do Decodificador.
        iii. O loop continua até que o modelo preveja um token de fim de sentença (`<EOS>`) ou atinja um limite de tamanho.

3.  **Decodificação:** A sequência de números (tokens) gerada é convertida de volta em palavras usando o vocabulário, formando a resposta final.

Este ciclo completo — desde a definição da arquitetura, passando pelo treinamento com dados, até a inferência em novas perguntas — encapsula o funcionamento fundamental de muitos LLMs modernos.
