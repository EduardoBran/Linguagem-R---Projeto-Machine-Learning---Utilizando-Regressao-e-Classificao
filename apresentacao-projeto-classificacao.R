# Estudo de Caso - Projeto BigDataNaPratica (Machine Learning) - Contexto

# Configurando o diretório de trabalho
setwd("C:/Users/Julia/Desktop/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/11-Projeto-BigDataNaPratica-Machine_Learning_em_Marketing_Digital-Prevendo_N_Usuarios_Convertidos")
getwd()


library(dplyr)
library(ggplot2)



######     Projeto Machine Learning em Marketing Digital - Prevendo a Probabilidade de Conversão do Lead      ###### 


## Contexto

# - Seu chefe convida você para uma reunião e informa que a área de Marketing Digital ficou muito satisfeita com seu projeto
#   de previsão do número de usuários convertidos. Mas agora, eles desejam um modelo que entregue uma previsão um pouco
#   diferente.

# - Considerando detalhes de uma nova campanha de Marketing, eles desejam saber se um lead (possível cliente) vai ou não
#   ser convertido (ou seja, se um potencial cliente vai ou não se tornar cliente e comprar o produto). Além disso, eles
#   desejam saber a probabilidade.

# -> Problema de negócio: Se a previsão do modelo é que um lead será convertido, qual a probabilidade de isso acontecer ?

# - Seu chefe descreve para você que esse é um projeto importante, de grande visibilidade para a área de Analytics e que
#   está um pouco preocupado se você tem as habilidades necessárias para conduzir o projeto e entregar o resultado.

# - Você diz ao seu chefe que concluir a formação Cientista de Dados e se sente apto e confortável ao novo desafio.

# - Você inicia o proejto, coleta a amostra de dados, aplica seu conhecimento para:
#
# -> interpretar os dados    
# -> fazer análise exploratória
# -> pré-processar os dados (que tem variáveis categórias),
# -> construir algumas versões do modelo e avaliar o modelo usando diferentes métricas
# -> ao final criar um procedimento para o deploy do modelo


# - Entretando, durante a analise dos dados, você irá se deparar com uma questão que o deixa incomodado. Uma das
#   variáveis do dataset aprensenta uma informação que tornaria o modelo de Machine Learning enviesado a uma
#   característica que poderia causar discriminação. O dilema estará claro e você precisa tomar sua decisão e justifica-la.

# - Você não poderá emitir uma opnião baseada em achismo, mas sim em dados. Precisará consultar a Lei Geral de Proteção de
#   Dados Pessoais (LGPD) e o Guia de Boas Práticas da LGPD a fim de identificar possíveis implicações judiciais do dilema
#   e então tomar a decisão sonre o que fazer com base nos dados e sob a luz da lei. Você então toma a decisãoe apreseta
#   solução.


# - O cenário acima é o tema do próximo projeto, que traz detalhes sobre a LGPD.




