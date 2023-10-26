# Exemplos uso Machine Learning 1

# Configurando o diretório de trabalho
setwd("C:/Users/Julia/Desktop/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/11-Projeto-BigDataNaPratica-Machine_Learning_em_Marketing_Digital-Prevendo_N_Usuarios_Convertidos")
getwd()


### EXEMPLOS


# Carregando pacotes

library(tidyverse) # manipulação de dados
library(dplyr)     # manipulação de dados
library(corrplot)  # criar gráfico de mapa de correlação
library(ggplot2)   # criar outros gráficos (especificamente de dispersão)
library(caret)     # usado em tarefas de classificação e regressão para simplificar o processo de treinamento de modelos
# install.packages("randomForest")
library(randomForest)

library(mlbench) # carrega dataset Sonar
library(MASS)    # carrega o dataset Pima
library(kernlab) # carrega o dataset spam


data(Pima.tr)
dados_Pima <- Pima.tr

data(Sonar)
dados_Sonar <- Sonar

data(spam)
dados_spam <- spam




###################   EXEMPLOS UTILIZANDO ALGORITMO DE REGRESSÃO LINEAR)   ###################   


##### EXEMPLO 1

# Contexto: Uma loja de jóias possui um relatório com todas as suas últimas 1000 ações de marketing feitas ao longo dos
#           últimos 2 anos. Sua ação de marketing constitui na contratação de de 1 até 5 pessoas fantasiadas na porta da 
#           loja. Cada ação de marketing teve um custo específico que varia entre 100 e 2000 reais dependendo da empresa e/ou
#           quantidade de pessoas contratadas. Durante cada ação era contabilizada as pessoas que entraram na loja e 
#           todas aquelas que de fato realizaram alguma compra. Precisamos um modelo de Machine Learning para conseguirmos
#           descobrir as próximas projeções de clientes que realizaram compras.

# Problema de Negócio: Criar um modelo que, ao recebeber novos dados, seja capaz de prever o número de usuários que efetuaram
#                      uma compra.

# Variáveis    : valor_gasto_campanha, funcionarios_contratado, pessoas_entraram, pessoas_compraram
# Variável alvo: pessoas_compraram

# Cria/carrega dados
set.seed(42)
num_acoes <- 1000

# Crie um data frame com as informações das ações de marketing
dados_loja <- data.frame(
  valor_gasto_campanha = round(runif(num_acoes, min = 100, max = 2000)),  # Custos variam de 100 a 2000 reais com 2 casas decimais
  funcionarios_contratados = sample(1:5, num_acoes, replace = TRUE),  # Contratação de 1 a 5 funcionários
  pessoas_entraram = round(rnorm(num_acoes, mean = 500, sd = 100)),  # Número médio de pessoas que entraram (distribuição normal)
  pessoas_compraram = integer(num_acoes)  # Inicialmente, defina o número de compras como zero
)

# Gere o número de compras com base nas pessoas que entraram
for (i in 1:num_acoes) {
  max_compraram <- min(dados_loja$pessoas_entraram[i], round(rnorm(1, mean = dados_loja$pessoas_entraram[i]/2, sd = 50)))
  # Ajuste para que pessoas_compraram siga uma distribuição normal entre 0 e max_compraram
  dados_loja$pessoas_compraram[i] <- max(0, min(round(rnorm(1, mean = max_compraram, sd = 10)), max_compraram))
}

# Visualize os primeiros registros dos dados
head(dados_loja)
View(dados_loja)


## Análise Exploratória dos Dados


# Verificando tipos de dados
str(dados_loja)

# Sumário estatístico
summary(dados_loja)

# Criando/calculando uma matriz de correlação
cor_matrix <- cor(dados_loja)
cor_matrix


# E para não ter que analisar somentes os números, vamos criar um gráfico corrplot

# Corrplot (para outras cores usar colors())
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         addCoef.col = 'springgreen2',
         tl.col = "black",
         tl.srt = 45)

# variável alvo "pessoas_compraram" tem forte correlação com a variável preditora "pessoas_entraram"
# não foi detectado problema de multicolinearidade


## Criando gráficos do tipo dispersão para visualizar a relação entre cada variável preditora com a variável alvo


# Scatter plot entre Valor Gasto em Campanha e Pessoas Compraram

ggplot(dados_loja, aes(x = valor_gasto_campanha, y = pessoas_compraram)) +
  geom_point(aes(color = valor_gasto_campanha), alpha = 0.6) +
  ggtitle("Scatter Plot entre Valor Gasto em Campanha e Pessoas Compraram") +
  xlab("Valor Gasto em Campanha") +
  ylab("Pessoas Compraram")


# Scatter plot entre Pessoas Entraram e Pessoas Compraram

ggplot(dados_loja, aes(x = pessoas_entraram, y = pessoas_compraram)) +
  geom_point(aes(color = pessoas_entraram), alpha = 0.6) +
  ggtitle("Scatter Plot entre Pessoas Entraram  e Pessoas Compraram") +
  xlab("Pessoas Entraram") +
  ylab("Pessoas Compraram")


# Scatter plot entre Funcionarios Contratados e Pessoas Compraram

ggplot(dados_loja, aes(x = funcionarios_contratados, y = pessoas_compraram)) +
  geom_point(aes(color = funcionarios_contratados), alpha = 0.6) +
  ggtitle("Scatter Plot entre Funcionarios Contratados e Pessoas Compraram") +
  xlab("Funcionarios Contratados") +
  ylab("Pessoas Compraram")




## Dividindo os dados em treino e teste
indices <- createDataPartition(dados_loja$pessoas_compraram, p = 0.80, list = FALSE)  

dados_loja_treino <- dados_loja[indices, ]
dados_loja_teste <- dados_loja[-indices, ]




##### Modelagem ##### 


# Versão 1 do Modelo - Regressão Linear Simples

# Residuals:
#      Min       1Q   Median       3Q      Max 
# -165.826  -33.826   -0.321   34.168  163.167 

# Coefficients:
#                 Estimate Std. Error t value Pr(>|t|)    
# (Intercept)      -4.64418    8.32474  -0.558    0.577    
# pessoas_entraram  0.49995    0.01636  30.559   <2e-16 ***

# Residual standard error: 51.56 on 998 degrees of freedom
# Multiple R-squared:  0.4834,	Adjusted R-squared:  0.4829 
# F-statistic: 933.8 on 1 and 998 DF,  p-value: < 2.2e-16

modelo_v1 <- lm(data = dados_loja_treino, pessoas_compraram ~ pessoas_entraram)
summary(modelo_v1)

# - o modelo de regressão linear simples sugere que o número de pessoas que entram na loja é um preditor significativo
#   do número de pessoas que realizam compras, e aproximadamente 48.34% da variabilidade na variável "pessoas_compraram"
#   pode ser explicada por essa variável preditora. No entanto, o modelo não é uma ótima combinação para explicar
#   completamente a variabilidade na variável alvo, uma vez que outros fatores não incluídos no modelo também podem 
#   desempenhar um papel importante.




# Versão 2 do Modelo - Regressão Linear Múltipla

# Residuals:
#      Min       1Q   Median       3Q      Max 
# -167.889  -33.661    0.399   34.319  163.045 

# Coefficients:
#                            Estimate Std. Error t value Pr(>|t|)
# (Intercept)               3.1961436  9.8002747   0.326    0.744
# valor_gasto_campanha     -0.0004382  0.0029452  -0.149    0.882
# funcionarios_contratados -2.1412594  1.1718202  -1.827    0.068 . 
# pessoas_entraram          0.4979231  0.0163905  30.379   <2e-16 ***

# Residual standard error: 51.52 on 996 degrees of freedom
# Multiple R-squared:  0.4851,	Adjusted R-squared:  0.4836 
# F-statistic: 312.8 on 3 and 996 DF,  p-value: < 2.2e-16

modelo_v2 <- lm(data = dados_loja_treino, pessoas_compraram ~ valor_gasto_campanha + funcionarios_contratados + pessoas_entraram)
summary(modelo_v2)

# - o modelo de regressão linear múltipla sugere que o número de pessoas que entram na loja é um preditor significativo 
#   do número de pessoas que realizam compras. No entanto, o valor gasto na campanha e o número de funcionários 
#   contratados não parecem ter um impacto significativo na variável alvo. O modelo explica aproximadamente 48.51% da
#   variabilidade na variável "pessoas_compraram." A significância da variável "funcionarios_contratados" não é 
#   conclusiva, pois seu valor p é 0.068. Pode ser importante, mas com alguma incerteza.



#### Interpretação de qual modelo usar

# - Como os valores de Residual standard error e Multiple R-squared dos dois modelos são muito próximos e com base nessas
#   interpretações, o Modelo V2 inclui variáveis que não são estatisticamente significativas (valor_gasto_campanha e 
#   funcionarios_contratados) e, portanto, é mais apropriado escolher o Modelo V1. Além disso, o Modelo V1 é mais simples,
#   envolvendo apenas uma variável independente, o que pode ser preferível em termos de interpretação e explicação.
# - Em resumo, o Modelo V1 (Regressão Linear Simples) é preferível para prever o número de pessoas que realizam compras
#   com base no número de pessoas que entram na loja, pois é mais simples e tem variáveis significativas. O Modelo V2
#   não oferece benefícios adicionais significativos e inclui variáveis não significativas.


#### Previsões

previsoes_dados_loja <- predict(modelo_v1, newdata = dados_loja_teste, type = 'response')
previsoes_dados_loja


#### Previsões com novos dados

# Criando novos dados para inserir nas previsões
novos_dados_loja <- data.frame(
  valor_gasto_campanha = c(1000, 1500),
  funcionarios_contratados = c(2, 2),
  pessoas_entraram = c(400, 500),
  pessoas_compraram = c(200, 250)
)


# Fazendo previsões
previsoes <- predict(modelo_v1, newdata = novos_dados_loja)
previsoes




## Interpretação final

# - No primeiro cenário, em que gastamos $1,000 em campanha e 400 pessoas entraram na loja, o modelo prevê que cerca de 194.23 
#   pessoas farão compras. No segundo cenário, com um aumento no gasto da campanha para $1,500 e 500 pessoas entrando na loja,
#   a previsão é de aproximadamente 244.92 pessoas comprando.






##### EXEMPLO 2

# Contexto: Você foi contratado para treinar um modelo de regressão para prever o consumo de combustível (milhas por galão)
#           de cada veículo com base em características como potência do motor, peso, número de cilindros, entre outras.
#           Isso permitirá identificar quais veículos são mais eficientes em termos de consumo de combustível e otimizar a 
#           alocação de veículos para rotas específicas.

# Problema de Negócio: Criar um modelo que, ao recebeber novos dados, seja capaz de prever o consumo de combustível
#                      (milhas por galão) de cada veículo com base em características como potência do motor, peso, número de 
#                      cilindros, entre outras.

# Variáveis    : 
# Variável alvo: mpg


# Carregando os dados
dados_mtcars <- mtcars


## Análise Exploratória

# verificando valores ausentes
colSums(is.na(dados_mtcars))

# Verificando tipos de dados
str(dados_mtcars)

# Sumário estatístico
summary(dados_mtcars)

# Criando/calculando uma matriz de correlação
cor_matrix <- cor(dados_mtcars)
cor_matrix

# Corrplot
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         addCoef.col = 'springgreen2',
         tl.col = "black",
         tl.srt = 45)


## Dividindo os dados em treino e teste
indices <- createDataPartition(dados_mtcars$mpg, p = 0.75, list = FALSE)  

dados_mtcars_treino <- dados_mtcars[indices, ]
dados_mtcars_teste <- dados_mtcars[-indices, ]



#### MODELAGEM ####


## Versão 1 do Modelo (Regressão Linear Múltipla) ####

# - utilizando todas as variáveis preditoras

modelo_v1 <- lm(data = dados_mtcars_treino, mpg ~ .)

summary(modelo_v1)


## Versão 2 do Modelo (Regressão Linear Simples) ####

# - utilizando somente a variável wt

modelo_v2 <- lm(data = dados_mtcars_treino, mpg ~ wt)

summary(modelo_v2)


## Versão 3 do Modelo (Regressão Linear Múltipla com seleção de variáveis) ####

# - utilizando todas as variáveis preditoras menos cyl, vs, am, gear e carb

modelo_v3 <- lm(data = dados_mtcars_treino, mpg ~ disp + hp + drat + wt + qsec)

summary(modelo_v3)

head(dados_mtcars_teste)


#### Interpretação de qual modelo usar

# modelo_v1: inclui múltiplas variáveis independentes para fazer previsões. Embora tenha um R-squared mais alto (0.869),
#            isso não significa necessariamente que seja o melhor modelo. Alguns coeficientes não são significativos e podem
#            estar introduzindo ruído no modelo.

# modelo_v2: é mais simples, prevendo o consumo de combustível (mpg) com base apenas no peso (wt). Embora seu R-squared seja
#            um pouco menor (0.7528), ele tem menos complexidade e apenas dois coeficientes significativos, tornando-o mais
#            interpretable.

# modelo_v3: obteve resultados muito bons e possui um número menor de variáveis, o que pode ser vantajoso em termos de
#            interpretabilidade.

# - A escolha do melhor modelo depende de seus objetivos. Se você valoriza um modelo com mais variáveis para obter um melhor
#   ajuste, o "modelo_v1" é uma opção. Se você deseja uma boa performance com menos variáveis, o "modelo_v3" pode ser
#   preferível. Recomenda-se também validar esses modelos com dados de teste antes de tomar uma decisão final.



#### Previsões

previsoes_dados_mtcars <- predict(modelo_v3, newdata = dados_mtcars_teste, type = 'response')
previsoes_dados_mtcars


#### Previsões com novos dados

# Criando novos dados para inserir nas previsões
novos_dados_mtcars <- data.frame(
  mpg = c(15, 20, 25),   # Valores de consumo de combustível (milhas por galão)
  cyl = c(8, 4, 6),      # Número de cilindros
  disp = c(350, 120, 200),  # Cilindrada
  hp = c(200, 80, 95),    # Potência do motor
  drat = c(3.5, 4.0, 3.9),  # Taxa de compressão do eixo traseiro
  wt = c(4.0, 2.5, 3.0),  # Peso do veículo
  qsec = c(15, 19, 20),  # Tempo para percorrer um quarto de milha
  vs = c(0, 1, 0),       # Motor em forma de V (0 = não, 1 = sim)
  am = c(0, 1, 1),       # Transmissão automática (0 = não, 1 = sim)
  gear = c(3, 4, 5),     # Número de marchas
  carb = c(4, 2, 1)      # Número de carburadores
)

# Editando dataset (Removendo variáveis)
novos_dados_mtcars <- novos_dados_mtcars[, !(names(novos_dados_mtcars) %in% c("cyl", "am", "vs", "gear", "carb"))]

# Visualizando o dataset resultante
novos_dados_mtcars

# Fazendo previsões com os novos dados
previsoes_novos_dados_mtcars <- predict(modelo_v3, newdata = novos_dados_mtcars)
previsoes_novos_dados_mtcars


#### Interpretação final

# - Com base nas características dos veículos fornecidas nos novos dados, o modelo de regressão prevê o seguinte consumo de
#   combustível (milhas por galão):
  
# - 1. Novo Dado 1: Estimativa de consumo de combustível de aproximadamente 14.64 mpg.
# - 2. Novo Dado 2: Estimativa de consumo de combustível de cerca de 25.36 mpg.
# - 3. Novo Dado 3: Estimativa de consumo de combustível de aproximadamente 23.56 mpg.

# Essas previsões podem ser úteis para avaliar a eficiência de cada veículo em termos de consumo de combustível.





##### EXEMPLO 2




















###################   EXEMPLOS UTILIZANDO ALGORITMO DE CLASSIFICAÇÃO LINEAR)   ###################   


##### EXEMPLO 1

# Contexto

# - Imagine que você trabalha para um instituto de pesquisa botânica que se dedica ao estudo e à preservação da flora em uma
#   área de conservação. Sua organização está interessada em automatizar a identificação de espécies de flores nativas para
#   melhor compreender a biodiversidade da região.
# - Nesse cenário, você pode usar o dataset Iris para criar um modelo de classificação que classifique automaticamente as
#   flores em diferentes espécies com base em suas características. Isso pode ser uma ferramenta valiosa para identificar
#   espécies de flores rapidamente, o que é fundamental para estudos de conservação.
# - Por exemplo, quando os pesquisadores coletam amostras de flores na área de conservação, eles podem medir características
#   como comprimento e largura das sépalas e pétalas. Em seguida, o modelo de classificação baseado no dataset Iris pode ser 
#   usado para determinar a espécie da flor com base nessas medidas.


# Problema de Negócio: Consiste em desenvolver um modelo de classificação, utilizando o dataset Iris, para automatizar a
#                      identificação de espécies de flores nativas em uma área de conservação. Isso permitirá aos pesquisadores
#                      acelerar o processo de classificação com base nas características das flores, contribuindo para um melhor
#                      entendimento da biodiversidade da região e apoiando esforços de conservação.


# Carregando dataset
dados_iris <- iris
head(iris)

# Variável alvo: "Species" é a variável que contém as classes ou categorias nas quais as flores são classificadas.
#                Portanto, "Species" é a variável que você deseja prever ou classificar com base nas outras características 
#                do dataset.


#### Análise Exploratória dos Dados


# verificando valores ausentes nas colunas
colSums(is.na(dados_iris))

# Verificando tipos de dados
str(dados_iris)

# Sumário estatístico (var Species já esta como do tupo factor)
summary(dados_iris)


# Dividindo os dados em treino e teste
indices <- createDataPartition(dados_iris$Species, p = 0.75, list = FALSE)

dados_iris_treino <- dados_iris[indices, ]
dados_iris_teste <- dados_iris[-indices, ]



##### Modelagem Preditiva #####

## Versão 1 (usando a Máquina de Vetores de Suporte (SVM))

modelo_v1 <- train(Species ~ ., data = dados_iris_treino, method = "svmRadial")
modelo_v1

# Fazer previsões no conjunto de teste
previsoes <- predict(modelo_v1, newdata = dados_iris_teste)
previsoes

# Avaliar o desempenho do modelo
confusao <- confusionMatrix(previsoes, dados_iris_teste$Species)
print(confusao)







#### Exemplo 2













###################   EXEMPLOS UTILIZANDO randomForest)   ###################   


##### EXEMPLO 1 (utilizando randomForest)

# Contexto: exemplo de dados hipotéticos de jogadores de futebol e, em seguida, demonstrar como você pode criar um modelo de
#           Machine Learning para prever as chances de lesões com base nessas informações.


# Criando o dataset

# Definir uma semente para reprodutibilidade
set.seed(123)

# Gerar nomes de jogadores aleatórios (apenas para fins de demonstração)
nomes_jogadores <- c("Messi", "Ronaldo", "Neymar", "Mbappé", "Salah", "Kane", "Hazard", "Lewandowski",
                     "De Bruyne", "Benzema", "Fernandes", "Suárez", "Lukaku", "Sterling", "Griezmann",
                     "Kroos", "Modric", "Van Dijk", "Silva", "Kante", "Pogba", "Laporte", "Courtois",
                     "Mané", "Casemiro", "Alaba", "Sancho", "Di María", "Marquinhos")

# Criar um conjunto de dados hipotético com nomes de jogadores
dados_jogadores <- data.frame(
  NomeJogador = sample(nomes_jogadores, 30, replace = TRUE),
  Idade = sample(20:35, 30, replace = TRUE),
  JogosUltimaTemporada = sample(10:40, 30, replace = TRUE),
  MinutosJogados = sample(500:3000, 30, replace = TRUE),
  HistoricoLesoes = sample(0:5, 30, replace = TRUE),   # Número de lesões prévias
  TreinosSemanais = sample(2:7, 30, replace = TRUE),   # Número de treinos por semana
  ChancesLesao = runif(30, min = 0, max = 1)           # Variável de destino (definidas aleatoriamente)
)

# Visualizar as primeiras linhas dos dados
head(dados_jogadores)
View(dados_jogadores)



##### Modelagem ##### 

## Versão 1 do Modelo - (Random Forest - Floresta Aleatória)


# - Dividindo os dados em conjuntos de treinamento e teste.
# - Os dados iniciais dos jogadores serão divididos em dois conjuntos: treinamento e teste. O conjunto de treinamento é usado para
#   treinar o modelo, enquanto o conjunto de teste é usado para avaliar o desempenho do modelo.

# amostra aleatória de 20 jogadores de um total de 30. Isso significa que 20 jogadores serão usados para treinar o modelo.
amostra <- sample(1:30, size = 20, replace = FALSE) 

# O conjunto de treinamento é criado, contendo os dados dos 20 jogadores amostrados aleatoriamente.
dados_treinamento <- dados_jogadores[amostra, ]

# O conjunto de teste é criado, contendo os dados dos jogadores que não foram selecionados para o treinamento.
dados_teste <- dados_jogadores[-amostra, ]

View(dados_treinamento)
View(dados_teste)


# Treinar um modelo de floresta aleatória para prever as chances de lesão

# - A variável alvo é ChancesLesao, e as variáveis preditoras incluem Idade, JogosUltimaTemporada, MinutosJogados, HistoricoLesoes
#   e TreinosSemanais. O modelo é treinado usando os dados do conjunto de treinamento.

modelo_lesao <- randomForest(ChancesLesao ~ Idade + JogosUltimaTemporada + MinutosJogados +
                               HistoricoLesoes + TreinosSemanais, data = dados_treinamento)

# Fazer previsões no conjunto de teste
previsoes <- predict(modelo_lesao, newdata = dados_teste)

# Comparar as previsões com as chances reais de lesão
resultado <- data.frame(Real = dados_teste$ChancesLesao, Previsao = previsoes)
print(resultado)


#### Interpretação Final

# - Vamos pegar o jogador "Casemiro" como exemplo:
  
# -> O valor real das chances de lesão para Casemiro é 0.3219765.

# - O modelo de Machine Learning fez uma previsão de 0.3999727 para as chances de lesão de Casemiro.
# - Agora, interpretando esses números para Casemiro:
  
# -> O valor real representa a taxa ou probabilidade real das chances de lesão para Casemiro, com base em suas características
#    (idade, histórico de lesões, etc.).

# - A previsão feita pelo modelo representa a estimativa das chances de lesão que o modelo gerou com base nas informações
#   de Casemiro.
# - Nesse caso, o modelo previu um valor ligeiramente superior às chances reais de lesão de Casemiro. Isso pode indicar que o
#   modelo está sendo um pouco pessimista em relação às chances de lesão desse jogador.

# - O mesmo raciocínio se aplica a todos os outros jogadores da lista. O modelo faz previsões com base nas características de cada
#   jogador e as compara com as chances reais de lesão. O objetivo é que as previsões se aproximem o mais possível das chances 
#   reais.
















################# DATASETS QUE PODEMOS USAR ######################

#   O R é fornecido com vários datasets de exemplo. Alguns dos datasets mais comuns que já vêm instalados com o R incluem:

# iris: Um conjunto de dados famoso para classificação de flores em três espécies com base em características como comprimento
# da sépala, largura da sépala, comprimento da pétala e largura da pétala.

# mtcars: Um conjunto de dados que contém informações sobre diferentes modelos de carros, incluindo características como consumo
# de combustível, número de cilindros, cavalos de potência e muito mais.

# swiss: Um conjunto de dados que contém informações sobre medidas socioeconômicas de diferentes cantões suíços.

# faithful: Este conjunto de dados contém informações sobre as erupções do gêiser "Old Faithful" no Parque Nacional de
# Yellowstone.

# ChickWeight: Dados sobre o peso de pintinhos em diferentes grupos de tratamento em um experimento de crescimento de galinhas.

# Esses são apenas alguns dos conjuntos de dados que estão disponíveis com a instalação padrão do R.


# -> Para estudar a aplicação de modelos de regressão para Machine Learning, você pode usar os conjuntos de dados:

# - mtcars: contém informações sobre vários modelos de carros, incluindo características como consumo de combustível, número
#   de cilindros, cavalos de potência, entre outros. Pode ser usado para prever o consumo de combustível com base em outras
#   variáveis.
# - ChickWeight: prever o peso de pintinhos com base em diferentes tratamentos
# - swiss: contém informações sobre várias variáveis socioeconômicas e de saúde em diferentes cantões da Suíça. Você pode
#   usá-lo para aplicar modelos de regressão para prever variáveis como taxa de fertilidade com base em variáveis independentes.
# - airquality:  Ele inclui variáveis como temperatura, vento, radiação solar e níveis de ozônio, e é frequentemente usado para
#   prever os níveis de ozônio com base em outras variáveis meteorológicas.

# -> Para estudar a aplicação de modelos de classificação para Machine Learning, você pode usar conjuntos de dados:

# - iris: especialmente popular para problemas de classificação, onde o objetivo é classificar flores em diferentes espécies
#   com base em suas características.
# - Sonar: contém dados de sinais sonares enviados e refletidos por minas e rochas no oceano. O objetivo é classificar se o
#   objeto é uma mina ou uma rocha com base nas características dos sinais.
# - Pima.tr: contém informações sobre pacientes do sexo feminino de uma tribo indígena Pima no Arizona. O objetivo é classificar
#   se um paciente desenvolverá diabetes dentro de cinco anos com base em variáveis de saúde, como idade, número de gravidezes,
#   glicose, pressão sanguínea e outros.
# - 

# obtem lista com todos datasets

data()




install.packages("kernlab")

library(kernlab)

data(spam)











