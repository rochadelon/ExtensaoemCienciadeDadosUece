Repositório de Fundamentos em Ciência de Dados
(C-Jovem | Universidade Estadual do Ceará)

📊 Descrição
Repositório dedicado aos projetos e práticas desenvolvidos durante o Módulo Intermediário de Ciência de Dados. Aqui você encontrará exemplos de técnicas, algoritmos e aplicações em áreas como visualização de dados, machine learning e processamento de linguagem natural (NLP).

📂 Estrutura do Repositório
Cada pasta corresponde a um tópico estudado, contendo:

Códigos-fonte (Jupyter Notebooks ou scripts em Python).

Datasets utilizados (ou links para acesso).

Documentação explicativa dos projetos.

Tópicos Abordados
Fundamentos de Visualização de Dados

Criação de gráficos com Matplotlib e Seaborn.

Análise exploratória de dados (EDA) e construção de dashboards.

Fundamentos de Mineração de Dados

Padrões de associação com algoritmo Apriori (ex: análise de cesta de compras).

Modelagem de Dados

Pré-processamento: tratamento de dados ausentes, One-Hot Encoding, Label Encoder.

Programação Aplicada à Inteligência Artificial

Classificação de imagens (ex: cães vs. gatos) com redes neurais (Xception).

Aprendizado Supervisionado

Parte I: Regressão Linear para previsão de preços de imóveis.

Parte II: Regressão Logística para classificação de spam.

Aprendizado Não Supervisionado

Parte I: Clusterização com K-Means (segmentação de clientes).

Parte II: Redução de dimensionalidade com PCA.

Processamento Textual

Parte I: Análise de sentimentos com Naive Bayes + TF-IDF.

Parte II: Chatbot adaptativo usando RNN e Word2Vec.

🛠️ Tecnologias Utilizadas
Linguagens: Python

Bibliotecas: Pandas, NumPy, Scikit-learn, TensorFlow/Keras, NLTK, SpaCy.

Ferramentas: Jupyter Notebook, Google Colab, VS Code.

Python
Scikit-learn

🚀 Como Executar os Projetos
Clone o repositório:

bash
Copy
git clone https://github.com/seu-usuario/nome-do-repositorio.git  
Instale as dependências:

bash
Copy
pip install -r requirements.txt  
Abra os Jupyter Notebooks da pasta desejada e execute célula por célula.

📌 Exemplos de Projetos
Análise de Sentimentos:

python
Copy
from sklearn.naive_bayes import MultinomialNB  
from sklearn.feature_extraction.text import TfidfVectorizer  

# Código simplificado para análise de sentimentos  
vectorizer = TfidfVectorizer()  
X = vectorizer.fit_transform(textos)  
modelo = MultinomialNB().fit(X, rotulos)  
Clusterização com K-Means:
Exemplo de Gráfico de Clusterização

📄 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.

📧 Contato
Feito com ❤️ por [Seu Nome]

LinkedIn:  https://www.linkedin.com/in/delonrocha/

Email: alandelonsrocha@gmail.com
