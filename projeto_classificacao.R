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


# Tipos dos dados
str(dados)

# Sumário (variáveis numéricas)
summary(dados)


## Gráficos (para analisar como os dados estão organizados)


# Gráfico de barras

# - Exibe a distribuição da variável alvo "converte" (diz a qtd de sim e a qtd de não)
# - Podemos constatar que o "SIM" tem uma quantidade maior que o "NÃO", porém como podemos ver no grafico, a diferença não é
#   tão grande. Se a diferença fosse muito alto, teríamos que aplicar uma técnica de 'balanceamento de classe' antes de treinar
#   o modelo.

ggplot(dados, aes(x = converteu)) +
  geom_bar(aes(fill = converteu), alpha = 0.7) +
  geom_text(stat = 'count', aes(label = after_stat(count), vjust = -0.5)) +  # exibe o valor da qtd
  ggtitle("Distribuição da Variável 'Converteu'") +
  xlab("Converteu") +
  ylab("Quantidade")


# - Exibe a distribuição da variável "faixa_etaria" (diz a qtd de cada faixa etária)

ggplot(dados, aes(x = faixa_etaria)) +
  geom_bar(fill = "orangered3", alpha = 0.7) +
  geom_text(stat = 'count', aes(label = after_stat(count), vjust = -0.5)) +  # exibe o valor da qtd
  ggtitle("Distribuição de Faixa Etária") +
  xlab("Faixa Etária") +
  ylab("Quantidade")



# Boxplot

# - Olhando para caixa da classe do SIM podemos obsrevar uma grande quantidade de valores (numero_cliques) dentro da caixa,
#   o que indica que provavelmente temos uma relação de numero_cliques e se converteu ou não.
# - Olhando para caixa da classe NÃO, podemos constatar que a mediana (linha no meio) é muito menor do que a mediana da caixa 
#   do SIM, o que indica claramente que o numero_cliques tem uma relação direta com o fato do usuária converter ou não.
# - Pontos fora da caixa do NÃO indicam valores outliers.

ggplot(dados, aes(x = converteu, y = numero_cliques, fill = converteu)) +
  geom_boxplot() +
  ggtitle("Boxplot - Número de Cliques por Conversão") +
  xlab("Converteu") +
  ylab("Número de Cliques")


# Gráfico de Dispersão

# - Gráfico que mostra a relação entre duas variáveis numéricas (quantitativas)
# - Acrescentamos também mais informações ao gráfico como o acréscimo dos dados da variável converteu
# - E também foi adicionado a linha azul de confiança juntamente com intervalo de confiança (método usado lm)
# - Observando o gráfico podemos ver que quando aumenta o numero_acessos, aumenta o numero_cliques

ggplot(dados, aes(x = numero_acessos, y = numero_cliques)) +
  geom_point(aes(color = converteu), alpha = 0.6) +
  geom_smooth(method = 'lm') +
  ggtitle("Relação entre Número de Acessos e Número de Cliques") +
  xlab("Número de Acessos") +
  ylab("Número de Cliques")



## Sumarizar dados para obter a média do número de acessos por cidade

dados_sumarizados <- aggregate(numero_acessos ~ cidade, data = dados, FUN = mean)
dados_sumarizados
  
dados_suma <- 
  dados %>% 
  group_by(cidade) %>% 
  summarize(numero_acessos = mean(numero_acessos))
dados_suma  


# Gráfico de Barras com dados sumarizados

# - Exibe a média do número de acesso por cidade

ggplot(dados_sumarizados, aes(x = reorder(cidade, -numero_acessos), y = numero_acessos)) + # reorder
  geom_bar(stat = 'identity', aes(fill = cidade), alpha = 0.7) +
  ggtitle("Gráfico de Barras - Média do Número de Acessos por Cidade") +
  xlab("Cidade") +
  ylab("Média do Número de Acessos") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))











