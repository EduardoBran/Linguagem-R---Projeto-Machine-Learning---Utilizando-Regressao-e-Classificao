# Estudo de Caso - Projeto BigDataNaPratica (Machine Learning)

# Configurando o diretório de trabalho
setwd("C:/Users/Julia/Desktop/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/11-Projeto-BigDataNaPratica-Machine_Learning_em_Marketing_Digital-Prevendo_N_Usuarios_Convertidos")
getwd()




# Carregando pacotes

library(tidyverse) # manipulação de dados
library(dplyr)     # manipulação de dados
library(corrplot)  # criar gráfico de mapa de correlação
library(ggplot2)   # criar outros gráficos (especificamente de dispersão)
library(caret)     # usado em tarefas de classificação e regressão para simplificar o processo de treinamento de modelos




##########       Projeto Machine Learning em Marketing Digital - Prevendo Número de Usuários Convertidos       ########## 

# -> Utilizando Classificação Linear Múltipla

# -> Pergunta de Negócio: Um lead será convertido? Sim ou não? Qual a probabilidade?
#    (diferente do projeto regressão, agora não queremos mais saber quantos usuários serão convertidos, queremos apenas
#     saber se vai ser convertido, sim ou não e a probabilidade)

# - O fato do "sim ou não" está no problema de negócio indica que iremos prever uma classe ou categoria e por isso
#   usaremos um algoritimo de Classificação.

# - Podemos perceber também que a pergunta de negócio pede a probabilidade e a probabilidade é um número.
#   Por que não usar regressão?
#   Aqui entra o "detalhe" da Classificação, iremos usar um algoritimo que vai permitir a previsão do "sim ou não", 
#   entretanto este algoritimo entrega uma informação a mais que é justamente a probabilidade disso acontecer.

# - Portanto quando for feita a previsão, iremos dizer:
#   "com base nessas características, esse lead vai ou não converter e isso vai acontecer com X% de probabilidade".


## Carrega o dataset original

dados <- as.data.frame(read_csv("dataset_classificacao.csv"))
head(dados)
View(dados)

# - Observando os dados, podemos constatar que cada linha é um lead (possível usuário) com seus dados pessoais e com a 
#   informação se comprou ou não (variável "converteu").

# - Como podemos perceber, temos uma variável chamada "cor_da_pele" e isso implica em uma questão moral:
#   "podemos prever se o usuário vai ou não comprar o produto considerando a pele do usuário?". Não podemos.
# - Para este modelo e para não criarmos um modelo discriminatório, foi decidido com base na lei geral de proteção
#   dados que não iremos utilizar esta variável e iremos remove-la.

# - E quanto a questão da variável faixa etária? Podemos utilizar?
#   Se ao invés de faixa etária, tivéssemos a idade exata poderíamos também criar um modelo discriminatório, porém
#   não é o caso. (a dica é que quando se tem a variável com idade exata, convertar a variável idade para faixa_etaria)


## Análise Exploratória dos Dados

