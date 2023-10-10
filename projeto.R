# Estudo de Caso - Projeto BigDataNaPratica (Machine Learning)

# Configurando o diretório de trabalho
setwd("C:/Users/Julia/Desktop/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/11-Projeto-BigDataNaPratica-Machine_Learning_em_Marketing_Digital-Prevendo_N_Usuarios_Convertidos")
getwd()



##########       Projeto Machine Learning em Marketing Digital - Prevendo Número de Usuários Convertidos       ########## 



# Carregando pacotes

library(tidyverse) # manipulação de dados
library(dplyr)     # manipulação de dados
library(corrplot)  # criar gráfico de mapa de correlação
library(ggplot2)   # criar outros gráficos (especificamente de dispersão)



# Carrega o dataset (código da criação do dataset ao final)

dados <- read_csv("dataset_criado.csv")
head(dados)
View(dados)









































set.seed(200)
# Definir o tamanho do conjunto de dados
n <- 500

# Criar o conjunto de dados
dados <- data.frame(
  valor_gasto_campanha = as.integer(rnorm(n, mean = 1000, sd = 200)),
  numero_visualizacoes = as.integer(rnorm(n, mean = 5000, sd = 1000)),
  numero_cliques = as.integer(rnorm(n, mean = 60, sd = 10)),
  usuarios_convertidos = integer(n)
)

# Garantir que usuarios_convertidos seja sempre menor que numero_cliques
for (i in 1:n) {
  usuarios_convertidos <- abs(round(rnorm(1, mean = 30, sd = 10)))
  while (usuarios_convertidos >= dados$numero_cliques[i]) {
    usuarios_convertidos <- abs(round(rnorm(1, mean = 30, sd = 10)))
  }
  dados$usuarios_convertidos[i] <- usuarios_convertidos
}

# Verificar se a restrição está satisfeita (deve retornar 0)
sum(dados$usuarios_convertidos >= dados$numero_cliques)

# Rescale os valores para atender aos limites especificados
dados$valor_gasto_campanha <- round((dados$valor_gasto_campanha - min(dados$valor_gasto_campanha)) / (max(dados$valor_gasto_campanha) - min(dados$valor_gasto_campanha)) * (1495 - 505) + 505)
dados$numero_visualizacoes <- round((dados$numero_visualizacoes - min(dados$numero_visualizacoes)) / (max(dados$numero_visualizacoes) - min(dados$numero_visualizacoes)) * (7528 - 2376) + 2376)
dados$numero_cliques <- round((dados$numero_cliques - min(dados$numero_cliques)) / (max(dados$numero_cliques) - min(dados$numero_cliques)) * (116 - 9) + 9)
dados$usuarios_convertidos <- round((dados$usuarios_convertidos - min(dados$usuarios_convertidos)) / (max(dados$usuarios_convertidos) - min(dados$usuarios_convertidos)) * (101 - 4) + 4)


# Corrigir "usuarios_convertidos" quando for maior ou igual a "numero_cliques"
dados$usuarios_convertidos[dados$usuarios_convertidos >= dados$numero_cliques] <- dados$numero_cliques[dados$usuarios_convertidos >= dados$numero_cliques] - 1

# Verificar se a restrição está satisfeita (deve retornar 0)
sum(dados$usuarios_convertidos >= dados$numero_cliques)



# Verificar se os limites de valores estão satisfeitos
summary(dados)

View(dados)

# Salva o dataset em formato CSV
# write.csv(dados, file = "dataset_criado.csv", row.names = FALSE)



