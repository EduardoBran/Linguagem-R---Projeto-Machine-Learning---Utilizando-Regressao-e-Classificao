# Estudo de Caso - Projeto BigDataNaPratica (Machine Learning)

# Configurando o diretório de trabalho
setwd("C:/Users/Julia/Desktop/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/11-Projeto-BigDataNaPratica-Machine_Learning_em_Marketing_Digital-Prevendo_N_Usuarios_Convertidos")
getwd()


library(dplyr)
library(ggplot2)



##########       Projeto Machine Learning em Marketing Digital - Prevendo Número de Usuários Convertidos       ########## 


## Contexto

#      Você foi contratado como Cientista de Dados por uma empresa que comercializa produtos digitais. A empresa trabalha diversas
#   estratégias de Marketing Digital e gostaria de ter um modelo de Machine Learning capaz de prever quantos usuários serão
#   convertidos (ou seja, quantas pessoas comprarão os produtos da empresa) após cada campanha. Conseguindo fazer a previsão, a
#   empresa pode ter uma idéia mais clara de quanto deve investir em cada campanha e o retorno esperado. Isso ajudará também no
#   planejamento da empresa para comercialização e entrega do seu produto digital, além do uso de ferramentas e mídicas sociais.

#      Dados históricos de campanhas passadas estão disponíveis e seu trabalho como Cientista de Dados é consutrir um modelo que,
#   ao recebeber novos dados, seja capaz de prever o número de usuários convertidos em uma campanha de Marketing Digital. Além disso,
#   o Gestor de Marketing precisa saber qual seria o aumento no número de usuários convertidos se aumentar em 1 unidade o valor gasto
#   em uma campanha.

#      Entretanto, os dados têm problemas (exatamente o que você encontrará no dia a dia) e você deve detectar esses problemas, decidir
#   a melhor estratégia para resolvê-los e então criar seu modelo. Pode ser necessário criar diferentes versões do modelo até chegar ao
#   modelo ideal. Quando chegar à versão ideal do modelo, você deve fornecer uma interpretação completa de como o modelo gera o resultado
#   final para que os gestores tenham mais confiança no uso do modelo.

#      Por fim, você deve fornecer uma forma de fazer o deploy do modelo e usá-lo imediatamente com novos dados.


## Usaremmos dados fictícios que representam dados reais.


## O objetivo é sair do zero. Desde a definição do problema de negócio até entrega com deploy.