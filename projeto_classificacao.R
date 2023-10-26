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


#### Carrega o dataset original ####

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


# Remove a variável
dados$cor_da_pele <- NULL



#### Análise Exploratória dos Dados ####


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


## Gráfico de Barras com dados sumarizados

# - Exibe a média do número de acesso por cidade

ggplot(dados_sumarizados, aes(x = reorder(cidade, -numero_acessos), y = numero_acessos)) + # reorder
  geom_bar(stat = 'identity', aes(fill = cidade), alpha = 0.7) +
  ggtitle("Gráfico de Barras - Média do Número de Acessos por Cidade") +
  xlab("Cidade") +
  ylab("Média do Número de Acessos") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))





##### Pré-Processamento e Encoding de Variáveis Categóricas #####

# - O que é Encoding de Variáveis Categóricas

# - Observando nosso dados podemos perceber que temos duas variáveis quantitativas e todas as outras variáveis são categóricas.
#   E lembrando que Machine Learning é matemática, como fazer matemática com palavras (variáveis categóricas) ?

# - Logo não podemos criar um modelo de Machine Learning com dados do tipo texto, por isso temos que codificar (encoding) estas
#   variáveis. Teremos que converter nossas variaveis categoricas para sua representação numérica. Ou seja, iremos transformar
#   o dado sem mudar a informação.

# - Exemplificando: qual o dado variável Cidade da primeira linha? A respota é conforme olhamos nos dados é Curitiba.
#   Então o que podemos fazer com a variável cidade ? Podemos substituir Curitiba por pelo nº 0, outra cidade pelo nº 1 e seguir
#   desta forma para os demais nomes de cidades. Com isso mudamos o dado sem perder a informação.

set.seed(42)

?createDataPartition

# - Antes de aplicarmos o encoding precisamos usar a função createDataPartition para poder particionar os dados e dividi-los em
#   amostras de treino e teste.
#   Quando aplicamos transformação ou pré processamento, em geral você faz isso com amostras diferentes de treino e teste.

# - O ideal é primeiro dividir em treino e teste e depois aplicar o processo de transformação, processamento e assim por diante.
#   Não é obrigatório. Em geral divide primeiro e depois aplica.

# - Iremos fazer a divisão com base na variável alvo, por que se não usar a variável alvo como critéria de divisão, poderiamos
#   colocar todos os registros de SIM em treino e todos de NAO em teste. E precisamos dividir de maneira aleatória.
#   Por isso foi usado a variável converteu como critério de separação.

# - p = 0.75 significa que 75% dos dados vão para a amostra de treino e o restante para amostra de teste. Sem formato de lista.

# Dividindo os dados em treino e teste
indices <- createDataPartition(dados$converteu, p = 0.75, list = FALSE)  

treino <- dados[indices, ]
teste <- dados[-indices, ]


# Visualiza os dados e tipos de dados
View(treino)
View(teste)
str(treino)
str(teste)


# -> Uma das formas de aplicar o encoding é utilizar a funcao as.factor()
#    Em python nao existe as.factor e vai ficar apenas o "0 ou 1". No R ele mantém o "texto".

# -> Em alguns algoritimos de Machine Learning não podemos utilizar apenas o as.factor e teremos que fazer a conversão de 
#    maneira explícita.


# Aplicando label encoding na variável alvo (converteu)
treino$converteu <- as.factor(treino$converteu)
teste$converteu <- as.factor(teste$converteu)

View(treino)
str(treino)
str(teste)


# Aplicando label encoding nas demais variáveis categóricas
treino$faixa_etaria <- as.factor(treino$faixa_etaria)
treino$cidade <- as.factor(treino$cidade)
treino$navegador_web <- as.factor(treino$navegador_web)

teste$faixa_etaria <- as.factor(teste$faixa_etaria)
teste$cidade <- as.factor(teste$cidade)
teste$navegador_web <- as.factor(teste$navegador_web)

View(treino)
str(treino)
str(teste)



##### Modelagem Preditiva #####


# -> Modelo utilizando Algoritmo de Regressão Logística (que é basicamente um algoritmo de classificação).
#    Ele entrega de fato uma previsão de classe.

# -> Importante não confundir Regressão Logística (prever classe/categoria) com Regressão Linear (prever valor numérico).


## Versão 1 (usando função glm para prever um valor numérico que ao final conseguiremos interpretar como classe/categoria)

modelo_v1 <- glm(data = treino, converteu ~ ., family = binomial(link = 'logit')) 

# lado esq do '~' utilizamos a variável alvo enquanto do lado direito do '~' colocamos as var preditoras.
# O '.' indica que vamos usar todas as variáveis do dataset.
# O family é a 'família' do glm(), neste caso utilizamos a 'binomial' pq ela entrega como resultado uma classe ou categoria.
# Estamos utilizando a 'família binomia' com o 'logit' porque é este 'logit' que entrega o número que irá sair na previsão.
# Poderíamos utilizar a 'multiclasse' quando entregamos várias classes.


summary(modelo_v1)

# Deviance Residuals: 
#     Min        1Q    Median        3Q       Max  
# -2.74429  -0.24418   0.00572   0.20703   2.26234  

# Coefficients:
#                      Estimate Std. Error z value Pr(>|z|)    
# (Intercept)          -8.15188    0.86818  -9.390  < 2e-16 ***
# numero_acessos        0.06397    0.00720   8.884  < 2e-16 ***
# numero_cliques        0.16320    0.01645   9.918  < 2e-16 ***
# faixa_etaria25-34     4.95394    0.61002   8.121 4.62e-16 ***
# faixa_etaria35-44     5.26812    0.64366   8.185 2.73e-16 ***
# faixa_etaria45-54     0.62731    0.48849   1.284    0.199    
# faixa_etaria55-64    -0.09947    0.46198  -0.215    0.830    
# cidadeCuritiba       -0.51948    0.46393  -1.120    0.263    
# cidadeFortaleza      -0.35657    0.42570  -0.838    0.402    
# cidadeNatal          -0.01710    0.44780  -0.038    0.970    
# cidadeSalvador        0.53860    0.43424   1.240    0.215    
# navegador_webEdge     0.59578    0.38128   1.563    0.118    
# navegador_webFirefox  0.28220    0.39775   0.709    0.478    
# navegador_webSafari   0.62325    0.39910   1.562    0.118    

# Null deviance: 1028.02  on 750  degrees of freedom
# Residual deviance:  340.81  on 737  degrees of freedom
# AIC: 368.81

# Number of Fisher Scoring iterations: 7

## Interpretação

# - Podemos notar que para prever a variável alvo 'converteu', poderemos usar 'numero_acessos', 'numero_cliques', 
#   'faixa_etaria25-34' e faixa_etaria35-44' pois estas tem um valor-p menor que 0.05.
#   As outras variáveis não são estatisticamente significante para prever a variável alvo.

# - Quanto menor o deviance residual em comparação com o deviance nulo, melhor o ajuste do modelo.

# - O AIC é um critério de informação usado para comparar modelos. Quanto menor o valor do AIC, melhor o ajuste do modelo.
#   Ele leva em consideração o ajuste do modelo e o número de variáveis no modelo.

# - Number of Fisher Scoring iterations: Indica o número de iterações necessárias para ajustar o modelo. Normalmente, um
#   número maior de iterações pode indicar que o modelo teve dificuldade em convergir.


## Fazendo Previsões no conjunto de teste
#  (O conjunto de treino é usado para treinar o modelo, enquanto o conjunto de teste é usado para avaliar o desempenho do modelo.)
previsoes_prob <- predict(modelo_v1, newdata = teste, type = 'response')
previsoes_prob

# Pegando os valores acima e realizando um ifelse indicando que se o valor for > 0.5 a previsao é 'sim', se não é 'não'
previsoes_classe <- ifelse(previsoes_prob > 0.5, 'sim', 'não')


## Criando a Matriz de confusão

# - Matriz de confusão é uma ferramenta fundamental na avaliação do desempenho de algoritmos de classificação em machine learning. 
#   Ela é criada para ajudar a entender o quão bem o modelo está fazendo previsões e classificando corretamente as amostras de dados
#   em diferentes classes. A matriz de confusão é especialmente útil quando se trabalha com problemas de classificação binária, 
#   onde você está tentando prever se algo pertence a uma de duas classes, como "sim" ou "não", "positivo" ou "negativo",
#   "spam" ou "não spam" entre outras.

matriz_confusao <- confusionMatrix(as.factor(previsoes_classe), teste$converteu)
matriz_confusao

#           Reference
# Prediction não sim
#        não  93  11
#        sim  15 130

#               Accuracy : 0.8956          
#                 95% CI : (0.8508, 0.9306)
#    No Information Rate : 0.5663          
#    P-Value [Acc > NIR] : <2e-16          

#                  Kappa : 0.7865          

# Mcnemar's Test P-Value : 0.5563          
                                          
#            Sensitivity : 0.8611          
#            Specificity : 0.9220          
#         Pos Pred Value : 0.8942          
#         Neg Pred Value : 0.8966          
#             Prevalence : 0.4337          
#         Detection Rate : 0.3735          
#   Detection Prevalence : 0.4177          
#      Balanced Accuracy : 0.8915          
                                          
#       'Positive' Class : não

## Interpretando

# Reference: O modelo fez 130 previsões corretas de "sim" (verdadeiros positivos).
#            O modelo fez 93 previsões corretas de "não".
#            Os valores ao lado são os erros. Logo uma diagonal são com os acertos e a outra diagonal são com os erros.

# Accuracy: taxa de erro (vai de 0 a 1), quanto for maior é melhor. Indica que ele está correto em 89,56% das previsões.

# Sensitivity (Sensibilidade): Também chamada de taxa de verdadeiros positivos, é a capacidade do modelo de identificar
# corretamente os casos positivos (leads que convertem). Neste caso, a sensibilidade é de 0.8611, ou seja, o modelo
# identifica corretamente 86,11% dos leads que convertem.

# Specificity (Especificidade): É a capacidade do modelo de identificar corretamente os casos negativos (leads que não 
# convertem). Neste caso, a especificidade é de 0.9220, indicando que o modelo identifica corretamente 92,20% dos leads que
# não convertem.

# 'Positive' Class : As métricas na matriz de confusão, como sensibilidade, especificidade, valor preditivo positivo e valor
# preditivo negativo, são calculadas com base na classe "não converteram" como a classe positiva. 



## Métrica de Avaliação (mesmo valor encontrado em matriz_confusao)
acuracia <- sum(diag(matriz_confusao$table)) / sum(matriz_confusao$table)
acuracia


# - Até aqui...
#  -> foi criado a v1 do modelo no dataset 'teste' utilizando Regressão Logística e foi feita a interpretação do seu resultado..
#  -> foi criado a as previsões no dataset 'treino' utilizando o modelo v1.
#  -> foi criado a matriz de confusão para avaliar o desempenho e foi feita a interpretação do seu resultado.


## Agora vamos observar novamente o sumário do modelo_v1

summary(modelo_v1)

# - Observando o sumário de modelo_v1 constatamos que além das variáveis "numero_acessos" e "numero_cliques", duas classes/categorias
#   da variável "faixa_etaria" são significantes para modelo, enquanto outras 2 classes não são. E agora como aplicar codificação 
#   quando cada classe/categoria da variável "faixa_etaria" aparece como uma variável ?

# - O correto seria mantermos "faixa_etaria" no dataset? Sim ou não?

# - Usando como regra que sempre temos que manter no nosso modelo as variáveis significativas e remover as que não são e 
#   tecnicamente falando o correto seria manter apenas as 2 classes/categorias que são estatisticamente. Então a resposta seria sim.
#   Mas teríamos um novo problema, estaríamos tornando nosso modelo tendencioso (neste caso de acordo com a idade da pessoa).
#   Por isso a importância de sempre compreender os dados em que estamos trabalhando.

# - Logo aqui os mais prudente então é remover toda a variável "faixa_etaria".



## Versão 2 do modelo

# - Levando em consideração toda a interpretação do modelo v1, removemos "faixa_etaria"

modelo_v2 <- glm(data = treino, converteu ~ numero_acessos + numero_cliques, family = binomial(link = 'logit'))


summary(modelo_v2)

# Deviance Residuals: 
#      Min        1Q    Median        3Q       Max  
# -2.34791  -0.62952   0.07993   0.58762   2.13941  

# Coefficients:
#                 Estimate Std. Error z value Pr(>|z|)    
# (Intercept)    -3.027801   0.252469 -11.993  < 2e-16 ***
# numero_acessos  0.030439   0.004280   7.112 1.14e-12 ***
# numero_cliques  0.095377   0.009233  10.330  < 2e-16 ***

# Null deviance: 1028.02  on 750  degrees of freedom
# Residual deviance:  597.12  on 748  degrees of freedom
# AIC: 603.12

# Number of Fisher Scoring iterations: 6


## Fazendo Previsões no conjunto de teste
previsoes_prob2 <- predict(modelo_v2, newdata = teste, type = 'response')
previsoes_classe2 <- ifelse(previsoes_prob2 > 0.5, 'sim', 'não')


## Matriz de confusão
matriz_confusao2 <- confusionMatrix(as.factor(previsoes_classe2), teste$converteu)
matriz_confusao2


## Métrica de Avaliação (mesmo valor encontrado em matriz_confusao)
acuracia2 <- sum(diag(matriz_confusao2$table)) / sum(matriz_confusao2$table)
acuracia2


# - Constatamos que o valor da accuracy diminuiu, ou seja, diminuímos a precisão do modelo para evitarmos um modelo tendencioso.




##### Deploy #####

# Salva o modelo treinado em disco
save(modelo_v2, file = "modelo_v2_class.RData")

# Carrega o modelo do disco
load("modelo_v2_class.RData")


## Criando novos dados para inserir nas previsões

# - iremos criar dados para realizar a previsão com 60 acessos e 20 cliques.

novos_dados <- data.frame(numero_acessos = c(60, 100, 50), numero_cliques = c(20, 40, 4))
novos_dados


## Previsões com os novos dados

previsoes_novos_dados_prob <- predict(modelo_v2, newdata = novos_dados, type = 'response')
previsoes_novos_dados_prob

previsoes_novos_dados_classe <- ifelse(previsoes_novos_dados_prob > 0.5, 'sim', 'não')
previsoes_novos_dados_classe


## Mostrando as previsões de classe e probabilidade

novos_dados$Lead_Convertido <- previsoes_novos_dados_classe
novos_dados$Probabilidade <- previsoes_novos_dados_prob * 100

novos_dados

# Fim




















