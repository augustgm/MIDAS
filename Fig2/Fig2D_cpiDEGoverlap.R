#########################################
# Number of cPI3000 DEGs in MIDAS top predictions
#########################################
library(ggplot2)

# Read in data
plot_df = read.csv(header = T, stringsAsFactors = F, file = "Fig2D_CPI-DEGs_forPlot.csv")
pvals = read.csv(header = T, stringsAsFactors = F, file = "Fig2D_empiricalPvals_forPlot.csv")

plot_df$variable = factor(plot_df$variable, levels = c("Lung\n(n=485)", "Pan\n(n=805)", "Bladder\n(n=187)", "Renal\n(n=133)"))
pvals$variable = factor(pvals$variable, levels = levels(plot_df$variable))

# Plot
ggplot(data = plot_df, mapping = aes(x = variable, y = value, fill = variable)) +
  geom_col(data = data.frame(variable = c("Pan\n(n=805)", "Lung\n(n=485)", "Renal\n(n=133)", "Bladder\n(n=187)"),
                             value = c(deg_n_pan, deg_n_lung, deg_n_renal, deg_n_bladder)),
             mapping = aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(fill = "white") +
  geom_text(data = pvals, mapping = aes(x = variable, y = 54, size = 10, 
                                        label = paste0("empirical,\np = ", value))) +
  ylab("Number of DEGs") + xlab("Cancer type") +
  scale_fill_manual(values = c("Pan\n(n=805)" = "#EDC2A4", 
                               "Lung\n(n=485)" = "#F68B6B",  
                               "Renal\n(n=133)" = "#F3C986",
                               "Bladder\n(n=187)" = "#F1BB7F")) + 
  scale_colour_manual(values = c("Pan\n(n=805)" = "#EDC2A4",   
                                 "Lung\n(n=485)" = "#F68B6B",   
                                 "Renal\n(n=133)" = "#F3C986",
                                 "Bladder\n(n=187)" = "#F1BB7F")) + 
  theme_bw() +  theme(text = element_text(size = 20), legend.position = "none")
