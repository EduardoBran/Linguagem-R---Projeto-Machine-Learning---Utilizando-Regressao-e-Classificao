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
# Ou seja, o precisamos exatamente é prever o número de usuarios convertidos.
# As demais serão candidatas a variáveis pretidoras (não é obrigatório usar todas), ou seja, precisamos encontrar um modelo que use as
# outras 3 variáveis ou 1 delas para prever usuarios convertidos

# Caso o problema de negócio fosse prever o numero de cliques ao invés de usuarios_convertidos teria que ser outro projeto pois
# Machine Learning é algo específico para resolver um problema pontual. Nosso raciocínio será apenas para prever números de usuários
# convertidos. Se mudarmos a variável alvo, mudamos o problema de negócio e precisamos mudar o projeto.


# No projeto original a correlação entre usuarios_convertidos e as demais variáveis é próxima a 1, assim como a relação entre as
# variáveis preditoras também é próxima a 1 e a lógica abaixo é partir disso:

# Precisamos e queremos que a variável alvo tenha a maior correlação possível com as variáveis preditoras,
# Precisamos e queremos que entre as variáveis preditoras tenha a menor correlação possível.

# Precisamos que as variáveis preditoras tenha a menor correlação possível pois isso é um problema chamado multicolinearidade
# Se tivermos uma correlação entre duas variáveis preditoras, é como se tivesse duplicando a informação. E isso é ruim para treinar
# o modelo de Machine Learning, é um problema que precisa ser resolvido. Neste caso precisaremos tratar isso.

# E a solução provável será não usar estas variáveis em alguns modelos que faremos abaixo.





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






##########################                MODELAGEM                ##########################


#### Versão 1 do Modelo (Regressão Linear Múltipla) ####

# Usaremos Regressão Linear Múltipla (é usada quando temos diversas variáveis preditoras)
# Usamos Regressão porque queremos prever uma variável numérica, caso a variável fosse categórica usaríamos Classificação Linear Múltipla

modelo_v1 <- lm(data = dados, usuarios_convertidos ~ valor_gasto_campanha + numero_visualizacoes + numero_cliques)
modelo_v1

summary(modelo_v1)


### Interpretando os valores originais

# Residuals:
# Min      1Q       Median      3Q     Max 
# -21.188  -4.104   -0.114   4.166  18.420 

# Coefficients:
#                       Estimate Std. Error t value  Pr(>|t|)    
# (Intercept)          -1.563463  1.081645   -1.445    0.149
# valor_gasto_campanha  0.007797  0.015754    0.495    0.621    
# numero_visualizacoes -0.003505  0.003139   -1.117    0.265    
# numero_cliques        0.943882  0.042275   22.327   <2e-16 ***

# Residual standard error: 6.865 on 496 degrees of freedom

# Multiple R-squared:  0.8617,	Adjusted R-squared:  0.8609 

# F-statistic: 1030 on 3 and 496 DF,  p-value: < 2.2e-16


# -> Importante sempre notar o valor p "Pr(>|t|)" e verificar se é maior ou menor que 0.05.
#    Se for maior, NÃO tem relevância com a variável alto, se for menor tem relevância.
#    As "***" indicam que é uma variável significativa.

# -> Importante verificar valor do "Multiple R-squared" e "Adjusted R-squared" e esperar que o valor fique próximo a 1.
#    O valor próximo a 1 indica que o modelo é estatisticamente significativo para explicar a variável alvo.


# -> Residuals:
# - Os resíduos são as diferenças entre os valores observados e os valores previstos pelo modelo. 
# - Os quartis dos resíduos (Min, 1Q, Median, 3Q, Max) dão uma ideia da distribuição dos erros. 
# - Em geral, você gostaria que esses valores fossem distribuídos simetricamente em torno de zero, o que indica que o modelo
#   faz um bom trabalho na previsão.

# -> Coefficients:
# - (Intercept): Este é o valor da variável dependente (usuários convertidos) quando todas as variáveis independentes são zero.
# - O valor é -1.563, mas o valor-p associado é maior que 0,05, o que indica que o intercepto não é significativamente diferente 
#   de zero neste modelo.

# -> valor_gasto_campanha:
# - O coeficiente é 0.007797, mas o valor-p associado é 0.621, o que indica que essa variável não é estatisticamente significativa
#   na previsão de usuários convertidos, pelo menos neste modelo.

# -> numero_visualizacoes:
# - O coeficiente é -0.003505, com um valor-p de 0.265. Isso também sugere que a variável não é significativa.

# ->  numero_cliques:
# - O coeficiente é 0.943882, com um valor-p extremamente baixo (< 2e-16). Isso indica que essa variável é altamente significativa
#   na previsão de usuários convertidos.

# -> Residual standard error:
# - Este é uma medida da qualidade do modelo. Quanto menor, melhor o modelo. Neste caso, é 6.865.

# -> Multiple R-squared e Adjusted R-squared:
# - Estes são indicadores da "qualidade" do modelo em termos de sua capacidade de prever a variável dependente. 
#   Um valor mais próximo de 1 é geralmente melhor. 
#   Neste caso, eles são relativamente altos (0.8617 e 0.8609, respectivamente), o que é bom.

# -> F-statistic e p-value:
# - Um teste F é realizado para determinar se o modelo como um todo é significativo. 
#   O valor F é 1030 e o valor-p associado é muito baixo (< 2.2e-16), indicando que o modelo é significativo.


## Interpretação Final

# - O modelo parece fazer um bom trabalho na previsão de "usuários convertidos" (R-squared alto), constatamos que apenas a
#   variável "número de cliques" é estatisticamente significativa na previsão. Isso pode implicar que "número de cliques" é a
#   principal variável que você deve se concentrar para entender as conversões de usuários.

# - As outras variáveis (valor gasto em campanha e número de visualizações) não são significativas neste modelo, o que sugere
#   que elas podem não ser úteis para prever a variável dependente, ou que outros fatores podem estar em jogo,
#   como multicolinearidade.


# - Até aqui indica que o modelo é ótimo pois temos um ótimo valor de Multiple R-squared e Adjusted R-squared, mas como percebemos
#   anteriormente ao criar o gráfico de correlação, os dados originais tem problema de multicolienaridade (correlção forte entre
#   variáveis preditoras) e por conta disso o modelo terá muitos problemas ao ser alimentado com novos dados.

# - E por conta disso este não é o modelo ideal.

# - Existem várias abordagens para lidar com a multicolinearidade: Remover Variáveis Redundantes. Transformar Variáveis, 
#   Regularização ou Coleta mais dados.

# - No nosso caso, criaremos uma segunda versão do modelo olhando somente para a variável preditora "numero_cliques" junto com a 
#   nossa variável alvo ("usuarios_convertidos), afinal o problema de multicolinearidade está entre as variáveis preditoras.

# - E como no nosso relatório anterior apontou que a única variável relevante é "numero_cliques" isso é mais um indicativo para
#   criar esta nova versão. Iremos então criar um Modelo De Regressão Linear Simples.






#### Versão 2 do Modelo (Regressão Linear Simples) ####

modelo_v2 <- lm(data = dados, usuarios_convertidos ~ numero_cliques)
modelo_v2

summary(modelo_v2)



### Interpretando os valores originais

# Residuals:
#      Min       1Q    Median       3Q      Max 
# -21.7207  -4.1067   -0.1993   4.2421  20.3177 

# Coefficients:
#                Estimate Std. Error  t value Pr(>|t|)    
# (Intercept)    -2.99028    1.00606   -2.972   0.0031 **
# numero_cliques  0.81490    0.01482   54.973   <2e-16 ***
#  ---
#  Signif. codes:  
#  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Residual standard error: 6.93 on 498 degrees of freedom
# Multiple R-squared:  0.8585,	Adjusted R-squared:  0.8582 
# F-statistic: 3022 on 1 and 498 DF,  p-value: < 2.2e-16


# - Começamos olhando para o valor p da nossa variável preditora (numero_cliques) e constatamos que por conta do valor
#   muito abaixo, podemos dizer que a variável é estatisticamente significativa para explicar a variável alvo.

# - Olhamos também para os valores de Multiple R-squared e Adjusted R-squared e constamtos também que possuem um valor próximo
#   a 1 , que indica que o modelo 2 possui praticamente a mesma perfomance do modelo 1. E como explicar isso, afinal tiramos 2
#   variáveis ? A resposta é por conta da multicolinearidade entre as variáveis.






# -> E agora ? Tudo indica que o trabalho terminou. Mas podemos ainda melhorar.

# - Nós inicamos a criação de um modelo com 4 variáveis, aplicamos a versão 1 do modelo para detectarmos qual variável preditora era
#   estatisticamente significa e detectamos 1 variável significativa. Antes havíamos detectado que tínhamos problema de 
#   multicolinearidade e decidimos resolver criando uma versão 2 do modelo deixando apenas 2 variáveis (Regressão Linear Simples)

# - E apesar do modelo 2 ter uma boa performance , nós só temos 1 variável preditora e isso signifca que tudo se resume a prever 
#   usuários convertidos olhando apenas para numero de cliques.

# - Então basicamente o que temos até aqui é "tomador de decisão, basta olhar para o numero de cliques para prever quantos usuarios
#   são convertidos".

# - E com isso temos um modelo simples demais, aonde resumimos demais a questão dizendo que iremos prever numero de usuarios com base
#   no numero de cliques.


# -> E assim como a versão 2 está muito simples pois da mesma forma que ter muitas variáveis preditoras é um problema, ter poucas
#    também é um prolema.

# -> Vamos então resolver o problema da multicolinearidade e utilizar mais variáveis para o nosso modelo.

# -> A abordagem de incluir mais variáveis pode ajudar a capturar nuances e fatores adicionais que influenciam o número de usuários
#    convertidos, tornando o modelo mais completo. 






#### Versão 3 do Modelo (Aplicando Engenharia de Atributos Antes da Regressão Linear Múltipla) ####






# arquivo original

































### Interpretando os meus valores (versão 1 - Modelo Regressão Linear Múltilpla):

# Residuals:
#   Min      1Q  Median      3Q     Max 
# -49.426  -8.046   1.413   9.207  31.425 

# Coefficients:

#                       Estimate   Std. Error  t value   Pr(>|t|)    
# (Intercept)          29.6623155   5.4949585    5.398   1.05e-07 ***
# valor_gasto_campanha  0.0005439   0.0040756    0.133      0.894    
# numero_visualizacoes -0.0007781   0.0008069   -0.964      0.335    
# numero_cliques        0.3312877   0.0335229    9.882     <2e-16 ***

# Residual standard error: 12.92 on 496 degrees of freedom

# Multiple R-squared:  0.1699,	Adjusted R-squared:  0.1648 

# F-statistic: 33.83 on 3 and 496 DF,  p-value: < 2.2e-16


# -> Residuals:
# - Os resíduos representam as diferenças entre os valores observados e os valores previstos pelo modelo. Neste modelo,
#   os resíduos variam de -49.426 a 31.425, com quartis em -8.046, 1.413, 9.207. Isso sugere que os resíduos não estão distribuídos
#   de maneira perfeitamente simétrica em torno de zero, indicando que o modelo pode não ser tão preciso quanto o modelo anterior.


# -> Coefficients:
# - O coeficiente do intercepto (Intercept) é 29.6623155, com um valor-p extremamente baixo (1.05e-07), o que indica que o
#   intercepto é significativamente diferente de zero.

# - O coeficiente de "valor_gasto_campanha" é 0.0005439, com um valor-p de 0.894, o que indica que esta variável não é 
#    estatisticamente significativa na previsão de "usuarios_convertidos" neste modelo.

# - O coeficiente de "numero_visualizacoes" é -0.0007781, com um valor-p de 0.335, indicando que esta variável também não é 
#   estatisticamente significativa.

# - O coeficiente de "numero_cliques" é 0.3312877, com um valor-p muito baixo (< 2e-16), sugerindo que esta variável é altamente 
#   significativa na previsão de "usuarios_convertidos".

# -> Residual standard error:
# - O erro padrão residual é 12.92, que é maior em comparação com o modelo anterior. Isso indica que este
#   modelo tem um erro médio maior na previsão.

# -> Multiple R-squared e Adjusted R-squared:
# - O R-quadrado múltiplo é 0.1699, e o R-quadrado ajustado é 0.1648. Isso indica que o
#   modelo atual explica apenas cerca de 16,99% da variância na variável de destino "usuarios_convertidos". Portanto, este modelo
#   tem um poder de explicação muito menor do que o modelo anterior.

# -> F-statistic e p-value:
# - O teste F mostra se o modelo como um todo é significativo. Neste caso, o valor F é 33.83, e o valor-p
#   associado é extremamente baixo (< 2.2e-16), indicando que o modelo é estatisticamente significativo.

## Interpretação Final

#  -> Em resumo, este novo modelo (modelo_v1) possui uma capacidade de explicação muito menor (R-quadrado mais baixo) e menor
#     precisão (erro padrão residual mais alto) em comparação com o modelo anterior. Além disso, as variáveis "valor_gasto_campanha"
#     e "numero_visualizacoes" não são estatisticamente significativas na previsão de "usuarios_convertidos", enquanto 
#     "numero_cliques" é a única variável altamente significativa.




### Interpretando os meus valores (versão 2 - Modelo Regressão Linear Simples):

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -48.902  -8.225   1.117   9.104  31.768 

# Coefficients:
#                Estimate Std. Error t value Pr(>|t|)    
# (Intercept)    26.51385    2.21393   11.98   <2e-16 ***
# numero_cliques  0.33392    0.03326   10.04   <2e-16 ***
#  ---
#  Signif. codes:  
#  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Residual standard error: 12.9 on 498 degrees of freedom
# Multiple R-squared:  0.1683,	Adjusted R-squared:  0.1666 
# F-statistic: 100.8 on 1 and 498 DF,  p-value: < 2.2e-16


# Residuals: Os resíduos representam as diferenças entre os valores observados e os valores previstos pelo modelo. Eles variam de
# -48.902 a 31.768, com quartis em -8.225, 1.117, 9.104. A distribuição dos resíduos parece não ser perfeitamente simétrica, 
#  mas é aceitável.

# Coefficients: O coeficiente da interceptação (Intercept) é 26.51385, e o coeficiente para "numero_cliques" é 0.33392. 
# Ambos os coeficientes têm valores-p muito baixos, indicando que eles são estatisticamente significativos. O Intercept representa
# o valor da variável dependente (usuários convertidos) quando o número de cliques é zero, que é 26.51385. O coeficiente para
# "numero_cliques" (0.33392) sugere que, para cada aumento unitário no número de cliques, espera-se um aumento de aproximadamente
# 0.33392 no número de usuários convertidos.

# Residual standard error: O erro padrão residual é 12.9, que é uma medida da qualidade do modelo. Quanto menor esse valor, melhor 
# o modelo se ajusta aos dados.

# Multiple R-squared e Adjusted R-squared: O Multiple R-squared é 0.1683, e o Adjusted R-squared é 0.1666. Isso indica que o modelo 
# tem uma capacidade limitada de explicar a variância na variável dependente, dado que possui apenas uma variável preditora.
# A maior parte da variabilidade na variável dependente não é explicada por este modelo.

# F-statistic e p-value: O valor do F-statistic é 100.8, e o valor-p associado é muito baixo, indicando que o modelo é 
# estatisticamente significativo como um todo, apesar de sua capacidade limitada de explicar a variância.

# Em resumo, este modelo de regressão linear simples com "numero_cliques" como preditor mostra que essa variável é estatisticamente 
# significativa para prever o número de usuários convertidos. No entanto, o modelo ainda não é capaz de explicar uma grande parte 
# da variabilidade na variável dependente, indicando que outros fatores não incluídos no modelo podem desempenhar um papel importante
# na previsão dos usuários convertidos.



### Interpretando os meus valores (Versão 3 do Modelo (Aplicando Engenharia de Atributos Antes da Regressão Linear Múltipla):















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








