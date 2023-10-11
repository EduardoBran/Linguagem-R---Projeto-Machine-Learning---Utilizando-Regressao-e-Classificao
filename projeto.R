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

dados <- as.data.frame(read_csv("dataset_criado.csv"))
head(dados)
View(dados)



## Análise Exploratória dos Dados


# Verificando tipos de dados
str(dados)

# Sumário estatístico
summary(dados)            # (notar valores de median e mean pq caso estejam proximos indicam uma distribuição normal)
#                           (caso os dados sigam uma distribuição normal tomamos uma direção, caso não sigam tomamos outra direção)
#                           (perceber também sem tem discrepância nos valores min e max)


# Criando/calculando uma matriz de correlação
cor_matrix <- cor(dados)
cor_matrix


# E para não ter que analisar somentes os números, vamos criar um gráfico corrplot

# Corrplot (para outras cores usar colors())
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         addCoef.col = 'springgreen2',
         tl.col = "black",
         tl.srt = 45)

# Observando a escala vemos que a correlação de coeficiente vai de -1 até 1
# A correlação próxima de -1 indica forte correlação negativa (aumenta o valor de uma variável e diminui o valor de outra variável)
# E a correlação próxima de 1 indica forte correlação posiiva (aumenta o valor de uma variável e aumenta o valor de outra variável)
# Próximo de 0 indica que não há correlação

# Durante a preparação do projeto, nós temos que definir nossa variável alvo ou variável de estudo
# Olhando para as variáveis decidimos que nossa variável alvo é "usuarios_convertidos" pois é exatamente esta que queremos prever.
# Ou seja, queremos prever o número de usuarios convertidos
# As demais serão candidatas a variáveis pretidotas (não é obrigatório usar todas), ou seja, precisamos encontrar o modelo que use as
# outras 3 variáveis para prever usuarios convertidos

# Caso o problema de negócio fosse prever o numero de cliques ao invés de usuarios_convertidos teria que ser outro projeto pois
# Machine Learning é algo especifico para resolver o problema pontual. Nosso raciocíono será apenas para prever numeros de usuarios
# convetidos. Se mudarmos a variável alvo, mudamos o problema de negócio e mudamos o projeto.


# No projeto original a correlação entre usuarios_convertidos e as demais variáveis é próxima a 1, assim como a relação entre as
# variáveis preditoras também é próxima a 1 e a lógica abaixo é partir disso:

# Precisamos e queremos que a variável alvo tenha a maior correlação possível com as variáveis preditoras,
# Precisamos e queremos que entre as variáveis preditoras tenha a menor correlação possível.

# Precisamos que as variáveis preditoras tenha a menor correlação possível pois isso é um problema chamado multicolinearidade
# Se tivermos uma correlação entre duas variáveis preditoras, é como se tivesse duplicando a informação. E isso é ruim para treinar
# o modelo de Machine Learning, é um problema que precisa ser resolvido. Neste caso precisaremos tratar isso.

# E a solução vai ser 





## Criando gráficos do tipo dispersão para visualizar a relação entre cada variável preditora com a variável alvo


# Scatter plot entre Valor Gasto em Campanha e Usuários Convertidos

ggplot(dados, aes(x = valor_gasto_campanha, y = usuarios_convertidos)) +
  geom_point(aes(color = valor_gasto_campanha), alpha = 0.6) +
  ggtitle("Scatter Plot entre Valor Gasto em Campanha e Usários Convertidos") +
  xlab("Valor Gasto em Campanha") +
  ylab("Usuários Convertidos")


# Scatter plot entre Numero de Visualizações e Usuários Convertidos

ggplot(dados, aes(x = numero_visualizacoes, y = usuarios_convertidos)) +
  geom_point(aes(color = numero_visualizacoes), alpha = 0.6) +
  ggtitle("Scatter Plot entre Numero de Visualizações e Usários Convertidos") +
  xlab("Numero de Visualizações") +
  ylab("Usuários Convertidos")


# Scatter plot entre Numero de Cliques e Usuários Convertidos

ggplot(dados, aes(x = numero_cliques, y = usuarios_convertidos)) +
  geom_point(aes(color = numero_cliques), alpha = 0.6) +
  ggtitle("Scatter Plot entre Numero de Cliques e Usários Convertidos") +
  xlab("Numero de Cliques") +
  ylab("Usuários Convertidos")





































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








