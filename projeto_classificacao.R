# Estudo de Caso - Projeto BigDataNaPratica (Machine Learning)

# Configurando o diretório de trabalho
setwd("C:/Users/Julia/Desktop/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/11-Projeto-BigDataNaPratica-Machine_Learning_em_Marketing_Digital-Prevendo_N_Usuarios_Convertidos")
getwd()




# Carregando pacotes

library(tidyverse) # manipulação de dados
library(dplyr)     # manipulação de dados
library(corrplot)  # criar gráfico de mapa de correlação
library(ggplot2)   # criar outros gráficos (especificamente de dispersão)




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


## Análise Exploratória dos Dados


