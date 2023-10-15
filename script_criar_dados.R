# Script para criação de dados




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

#Residuals:
#   Min      1Q  Median      3Q     Max 
#-51.594  -7.962   0.570   8.655  33.968 

# Coefficients:
#                       Estimate Std. Error t value Pr(>|t|)
# (Intercept)          2.956e+01  4.256e+00   6.946 1.18e-11
# valor_gasto_campanha 2.622e-03  4.122e-03   0.636    0.525
# taxa_de_clique       1.081e+03  1.218e+02   8.873  < 2e-16

# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Residual standard error: 13.15 on 497 degrees of freedom
# Multiple R-squared:  0.138,	Adjusted R-squared:  0.1345 
# F-statistic: 39.78 on 2 and 497 DF,  p-value: < 2.2e-16














### script para criação dos dados

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



