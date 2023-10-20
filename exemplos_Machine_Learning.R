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
# install.packages("randomForest")
library(randomForest)



# EXEMPLO 1

# Contexto: Uma loja de jóias possui um relatório com todas as suas últimas 100 ações de marketing feitas ao longo dos
#           últimos 2 anos. Sua ação de marketing constitui na contratação de de 1 até 5 pessoas fantasiadas na porta da 
#           loja. Cada ação de marketing teve um custo específico que varia entre 100 e 2000 reais dependendo da empresa e/ou
#           quantidade de pessoas contratadas. Durante cada ação era contabilizada as pessoas que entraram
#           na loja e todas aquelas que de fato realizaram alguma compra. Gostaríamos de realizar um modelo de Machine Learning
#           para conseguirmos descobrir as próximas projeções de clientes que realizaram compras.

# Problema: Criar um modelo que, ao recebeber novos dados, seja capaz de prever o número de usuários que efetuaram uma compra.

# Variáveis:     valor_gasto_campanha, funcionarios_contratado, pessoas_entraram, pessoas_compraram
# Variável alvo: pessoas_compraram

# Cria/carrega dados
set.seed(42)
num_acoes <- 100

# Crie um data frame com as informações das ações de marketing
dados_loja <- data.frame(
  valor_gasto_campanha = sprintf("%.2f", runif(num_acoes, min = 100, max = 2000)),  # Custos variam de 100 a 2000 reais com 2 casas decimais
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





























# EXEMPLO 2 (utilizando randomForest)

# Contexto: exemplo de dados hipotéticos de jogadores de futebol e, em seguida, demonstrar como você pode criar um modelo de
#           Machine Learning para prever as chances de lesões com base nessas informações.


# Cria o dataset

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



## Criando o modelo

# Dividir os dados em conjuntos de treinamento e teste

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

# Visualizar o resultado
print(resultado)



## Interpretação Final

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


































# Carregue os dados de treinamento e teste (esses dados são hipotéticos e apenas um exemplo)
set.seed(123)  # Para reprodutibilidade
dados_treinamento <- data.frame(
  ID = 1:100,
  Idade = sample(18:70, 100, replace = TRUE),
  Renda = rnorm(100, mean = 3500, sd = 1000),
  Emprego = sample(c("Empregado", "Desempregado", "Autônomo"), 100, replace = TRUE),
  EstadoCivil = sample(c("Solteiro", "Casado", "Divorciado"), 100, replace = TRUE),
  Inadimplente = sample(0:1, 100, replace = TRUE)
)

dados_teste <- data.frame(
  ID = 101:150,
  Idade = sample(18:70, 50, replace = TRUE),
  Renda = rnorm(50, mean = 3500, sd = 1000),
  Emprego = sample(c("Empregado", "Desempregado", "Autônomo"), 50, replace = TRUE),
  EstadoCivil = sample(c("Solteiro", "Casado", "Divorciado"), 50, replace = TRUE)
)

# Treine um modelo de floresta aleatória
modelo <- randomForest(Inadimplente ~ Idade + Renda + Emprego + EstadoCivil, data = dados_treinamento, ntree = 100)

# Faça previsões no conjunto de teste
previsoes <- predict(modelo, newdata = dados_teste, type = "response")

# Avalie o desempenho do modelo (geralmente, você usaria métricas apropriadas para problemas de classificação)
tabela_confusao <- table(dados_teste$Inadimplente, previsoes)
taxa_acerto <- sum(diag(tabela_confusao)) / sum(tabela_confusao)
print(taxa_acerto)
