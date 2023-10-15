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



# Carrega o dataset original

dados <- as.data.frame(read_csv("dataset.csv"))
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
#    As "***" indicam que é uma variável significativa e que tem relevância.

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
# - O coeficiente é 0.007797, mas o valor-p associado é 0.621, o que indica que essa variável não é estatisticamente 
#   significativa na previsão de usuários convertidos, pelo menos neste modelo.

# -> numero_visualizacoes:
# - O coeficiente é -0.003505, com um valor-p de 0.265. Isso também sugere que a variável não é significativa.

# ->  numero_cliques:
# - O coeficiente é 0.943882, com um valor-p extremamente baixo (< 2e-16). Isso indica que essa variável é altamente 
#   significativa na previsão de usuários convertidos.

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

# - E por conta disso este não seria o modelo ideal.

# - Existem várias abordagens para lidar com a multicolinearidade: Remover Variáveis Redundantes. Transformar Variáveis, 
#   Regularização ou Coleta mais dados.

# - No nosso caso, criaremos uma segunda versão do modelo olhando somente para a variável preditora "numero_cliques" junto com a 
#   nossa variável alvo ("usuarios_convertidos), afinal o problema de multicolinearidade está entre as variáveis preditoras.

# - E como o relatório da versão 1 do modelo apontou que a única variável relevante é "numero_cliques" isso é mais um indicativo para
#   criar esta nova versão. Iremos então criar a versão 2 do modelo usando Regressão Linear Simples.






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
#   a 1 , que indica que o modelo 2 possui praticamente a mesma perfomance do modelo 1.
# - E como explicar isso, afinal tiramos 2 variáveis ?
#   A resposta é por que retiramos as variáveis preditoras que não eram usadas e resolvemos o problema da multicolinearidade 
#   entre as variáveis.






# -> E agora ? Tudo indica que o trabalho terminou. Mas podemos melhorar...

# - Nós inicamos a criação da versão 1 do modelo com 4 variáveis, aplicamos o modelo para detectarmos qual variável preditora era
#   estatisticamente significativa na previsão da variável alvo "usuários_convertidos" e detectamos que 1 das 4 variáveis era uma
#   variável significativa ("numero_cliques"). Porém, antes da criação da versão 1, ao calcular uma matrix de correlação havíamos
#   detectado que tínhamos problema de multicolinearidade e para resolver este problema decidimos criar uma versão 2 do modelo
#   deixando apenas 2 variáveis (variável alvo "usuarios_convertidos" e variáve preditora significativa "numero_cliques") e 
#   utilizar a técnica de Regressão Linear Simples.

# - E apesar do modelo 2 ter uma boa performance, nós só temos 1 variável preditora e isso signifca que tudo se resume a prever 
#   usuários convertidos olhando apenas para numero de cliques.

# - Então basicamente o que temos até aqui é "tomador de decisão, basta olhar para o numero de cliques para prever quantos usuarios
#   são convertidos".

# - E com isso temos um modelo simples demais, aonde resumimos demais a questão dizendo que iremos prever numero de usuarios com base
#   no numero de cliques.


# -> E assim constatamos que a versão 2 está muito simples pois da mesma forma que ter muitas variáveis preditoras é um problema,
#    ter poucas também é um prolema.

# -> Vamos então abordar o problema da multicolinearidade de uma outra forma e utilizar mais variáveis para o nosso modelo.

# -> A abordagem de incluir mais variáveis pode ajudar a capturar nuances e fatores adicionais que influenciam o número de usuários
#    convertidos, tornando o modelo mais completo. 






#### Versão 3 do Modelo (Aplicando Engenharia de Atributos Antes da Regressão Linear Múltipla) ####

# - Até aqui nossa primeira decisão foi criar a versão 1 do modelo (Regressão Linear Múltipla) sem fazer qualquer tipo de 
#   modificação nos dados. Com isso verificamos que a performance era boa (R-squared) e detectamos 1 variável significativa
#   ("numero_cliques") para com a nossa variável alvo ("usuarios_convertidos").

# - Porém como já havíamos detectado anteriormente o problema de multicolinearidade, decidimos criar uma versão 2 do modelo 
#   utilizando Regressão Linear Simples. E o que aconteceu foi que apesar do modelo também ter boa performance, ele ficou
#   simples demais. E assim tomamos mais uma decisão que é criar uma versão 3 do modelo resolvendo o problema da multicolinearidade
#   e utilizar mais variáveis preditoras.


# - Para resolver nosso problema de multicolinearidade aplicaremos a técnica Engenharia de Atributos, ou seja, iremos olhar para
#   nossas variáveis, faremos alguma transformação e/ou criar uma nova variável.

# - Como a multicolinearidade é quando temos uma correlação muilto alto entre duas variáveis preditoras e isso indica que elas 
#   representam a mesma informação, nós podemos juntar a informação destas duas variáveis e criar uma terceira variável.


# Criando a nova variável taxa_de_clique (criamos esta variável pois para a pessoa chegar no clique ela passa pela visualização)
head(dados)

dados$taxa_de_clique <- dados$numero_cliques / dados$numero_visualizacoes


# Veriricar se algum valor ficou igual a zero (sempre verificar quando realizar divisão de valores)
any(dados$taxa_de_clique == 0)


# Calculando uma nova matriz de correlação
cor_matrix <- cor(dados)
cor_matrix

# Corrplot
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         addCoef.col = 'springgreen2',
         tl.col = "black",
         tl.srt = 45)

# -> Criar a variável taxa_de_clique significa que não precisamos mais utilizar as variáveis "numero_visualizacoes" e "numero_cliques".
#    A informação destas duas variáveis agora está na variável "taxa_de_clique".

# -> Agora interpretando a correlação da variável "taxa_de_clique" com a outra variável preditora "valor_gasto_campanha", detectamos
#    que a correlação entre essas duas variáveis agora é 0.06 (original), o que é um bom número.

# -> Olhamos também para a correlação entre a variável "taxa_de_clique" com a nossa variável alvo "usuarios_convertidos e detectamos
#    que o valor é 0.4 (dados originais). Não é um valor tão alto, mas também não está tão próximo de zero, o que é razoável.



# Calculando versão 3 do modelo

modelo_v3 <- lm(data = dados, usuarios_convertidos ~ valor_gasto_campanha + taxa_de_clique)
modelo_v3

summary(modelo_v3)

### Interpretando os valores originais

# Residuals:
#      Min       1Q    Median       3Q     Max 
# -23.8286  -4.5071   -0.1693   4.2813  20.744 

# Coefficients:
#                        Estimate   Std. Error  t value  Pr(>|t|)
# (Intercept)           -4.785e+01   2.601e+00   -18.39  <2e-16 ***
# valor_gasto_campanha  5.105e-02    1.102e-03    46,34  <2e-16 ***
# taxa_de_clique        3.613e+03    1.868e+02    19.34  <2e-16 ***

#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Residual standard error: 7.336 on 497 degrees of freedom
# Multiple R-squared:  0.8418,	Adjusted R-squared:  0.8412 
# F-statistic: 1322 on 2 and 497 DF,  p-value: < 2.2e-16


# - O modelo explica aproximadamente 84,18% da variação em usuarios_convertidos, o que é bom.

# - O coeficiente para "valor_gasto_campanha" é 0,05105, isso significa que mantendo todas as outras variáveis constantes
#   o aumento de uma unidade em "valor_gasto_campanha" resultará em um aumento de 0,05105 unidades na nossa variável alvo
#   "usuarios_convertidos".
# - O coeficiente para taxa_de_clique é 3613, isso significa que mantendo todas as outras variáveis constantes
#   o aumento de uma unidade em "taxa_de_clique" resultará em um aumento de 3613 unidades na nossa variável alvo
#   "usuarios_convertidos".

# -> "5.105e-02": Isso significa 5.105 multiplicado por 10 elevado à potência de -02. O expoente negativo indica que você move a
#     vírgula decimal duas posições para a esquerda. Portanto, "5.105e-02" é igual a 0,05105.
# -> "3.613e+03": Isso significa 3.613 multiplicado por 10 elevado à potência de 3. O expoente positivo indica que você move a
#    vírgula decimal três posições para a direita. Portanto, "3.613e+03" é igual a 3613.

# - Todos as variáveis preditoras são significativos, com valores-p muito baixos
#   (ou seja, estão marcadas como "***" e valor p abaixo de 0.05)

# - O modelo é estatisticamente significativo, conforme indicado pelo valor-p próximo a zero para a estatística F (F-statistic)

# - Lembre-se de que essas são interpretações puramente estatísticas. 
#   A validade prática desses resultados deve ser avaliada no contexto do problema de negócio que você está tentando resolver.






### Vamos checar as suposições do modelo de regressão (todo e qualquer modelo de Machine Learning possuem suposições)

# Obter os resíduos do modelo de regressão linear (diferenças entre os valores observados e os valores previstos pelo modelo)
# Os resíduos são essenciais para a avaliação das suposições do modelo
residuals <- resid(modelo_v3)
residuals


# Gráfico de Resíduos vs Valores Ajustados (gráfico para ajudar a verificar a suposição de homocedasticidade. )

# - Gráfico que relaciona os resíduos com os valores ajustados pelo modelo. O eixo x representa os valores ajustados 
#   (previstos pelo modelo), enquanto o eixo y representa os resíduos.
# - O gráfico ajuda a verificar a suposição de homocedasticidade. Homocedasticidade significa que a variabilidade dos resíduos é
#   constante em todos os níveis dos valores ajustados. Em outras palavras, os erros não devem mostrar nenhum padrão claro em
#   relação aos valores previstos. O que se espera ver é uma nuvem de pontos que se espalha aleatoriamente ao redor de zero, sem
#   formar padrões em forma de funil ou cone.
# - A linha suave no gráfico é uma representação visual que ajuda a identificar tendências ou padrões nos dados. Neste caso, está
#   sendo usado o método 'loess' para ajustar a linha suave.

ggplot(dados, aes(x = predict(modelo_v3), y = residuals)) +
  geom_point() +
  geom_smooth(se = FALSE, method = 'loess') +
  ggtitle("Resíduos vs Valores Ajustados") +
  xlab("Valores Ajustados") +
  ylab("Resíduos")


# Gráfico Histograma dos Resíduos (gráfico para ajudar a verificar a normalidade dos resíduos)

# - Um histograma em forma de sino indica que os resíduos estão normalmente distribuídos, indicando que temos um 
#   bom modelo de regressão.

ggplot(dados, aes(x = residuals)) +
  geom_histogram(binwidth = 1, fill = 'blue', alpha = 0.7) +
  ggtitle("Histograma dos Resíduos") +
  xlab("Resíduos")


# Gráfico QQ-plot (este gráfico também ajuda a verificar a normalidade dos resíduos)

# - Pontos alinhados em torno da linha diagonal sugerem que os resíduos são normalmente distribuídos, indicando 
#   que temos um bom modelo de regressão.

ggplot(dados, aes(sample = residuals)) +
  geom_qq() +
  geom_qq_line() +
  ggtitle("QQ-Plot dos Resíduos") +
  xlab("Quantis Teóricos") +
  ylab("Quantis Amostrais")


# - Como nós anteriormente resolvemos o problema da multicolinearidade, aplicamos engenharia de atributos, isso levou aos
#   nossos três gráficos mostrarem que nosso modelo (v3) segue as suposições para a regressão, ou seja, é um modelo equilibrado.





### Deploy do Modelo







































## Explicando Componentes do Sumário

# Residuals:
# Esta seção mostra um resumo estatístico dos resíduos (diferença entre os valores observados e os 
# valores previstos pelo modelo).

# Min, 1Q, Median, 3Q, Max descrevem a distribuição dos resíduos. 
# Seu objetivo é que esses valores sejam distribuídos simetricamente em torno de zero. Nesse caso, parece que 
# a mediana está próxima de zero, o que é um bom sinal.

# Coefficients:
# Esta seção descreve os coeficientes do modelo de regressão.

# Estimate: A estimativa dos coeficientes. Por exemplo, para cada unidade de aumento no valor_gasto_campanha, 
# a variável usuarios_convertidos aumenta em média 0.05105 unidades, mantendo a taxa_de_clique constante.

# Std. Error: O erro padrão dos coeficientes, uma medida da variação dos coeficientes.

# t value: A estatística t, usada para testar a hipótese nula de que o coeficiente é igual a zero (sem efeito). 
# Um valor t alto pode indicar que a variável é significativa.

# Pr(>|t|): O valor-p associado à estatística t. Um valor muito baixo (< 0,05) indica que você pode rejeitar 
# a hipótese nula. Isso significa que o coeficiente é estatisticamente significativo para prever a variável alvo.

# Todos os coeficientes são altamente significativos (p-valor < 2e-16), indicando que ambos são importantes preditores da variável alvo.

# Outras Estatísticas:
# Residual standard error: É uma medida da qualidade do ajuste do modelo aos dados. Quanto menor, melhor, 
# embora deva ser interpretado no contexto do problema.

# Multiple R-squared e Adjusted R-squared: São medidas que indicam a proporção da variação na variável 
# dependente que é explicada pelo modelo. O seu valor é de 0,8418, o que é relativamente alto e indica um bom ajuste.

# F-statistic e p-value: Estas estatísticas testam a hipótese nula de que todos os coeficientes de regressão 
# são iguais a zero. Dado o valor extremamente baixo do valor-p, você pode rejeitar essa hipótese.
