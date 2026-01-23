########################
# Radar plot for PDE data
##########################
# radar plot for marker expression for CPM flow data
library(tidyverse)
library(fmsb)

rm(list = ls())


`%ni%` = Negate(`%in%`)

# define treatment to plot
des_cond = "PTPN22i"  # PTPN22i, aOSM

# define path  - data is %
des_sheet = c("GZMB_NonTregs", "HLA-DR_NonTregs", "Ki67_NonTregs", "OX40_NonTregs", 
              "41BB_CD8", "GZMB_CD8", "HLA-DR_CD8", "Ki67_CD8", "OX40_CD8", "PD1_CD8",  
              "Dysfunctional", "Predysfunctional")

# load in the data
rad = data.frame()
for (i in 1:length(des_sheet)) {
  
  if (des_sheet[i] %in% c("Dysfunctional", "Predysfunctional")) {
    curr_sheet = paste0("Fig5B_5D_", des_sheet[i])
    
  } else {
    curr_sheet = paste0("Fig5C_5E_", 
                        ifelse(test = (length(grep(pattern = "NonTregs", x = des_sheet[i])) > 0), 
                               yes = gsub(pattern = "NonTregs", replacement = "CD4_Teff", x = des_sheet[i]),
                               no = gsub(pattern = "CD8", replacement = "CD8_T", x = des_sheet[i])))
  }
  
  tmp <- readxl::read_excel(path = "Fig5_SourceData.xlsx", 
                            sheet = curr_sheet, na = "N/A") %>% as.data.frame()  
  colnames(tmp)[1] = "Patient_ID"
  tmp$marker = des_sheet[i]
  rad = rbind(rad, tmp)
}
rad = rad %>% filter(Patient_ID != "p")  # excluding p-values
str(rad)
rad$`PTPN-22i` = as.numeric(rad$`PTPN-22i`)
rad$aOSM = as.numeric(rad$aOSM)
unique(rad$marker)
colnames(rad)[colnames(rad) == "PTPN-22i"] = "PTPN22i"


#### 5B and 5D ####

## plot T cell subsets
tsub = rad %>% filter(marker %in% c("Dysfunctional", "Predysfunctional")) %>%
  reshape2::melt(id.vars = c("Patient_ID", "marker"), value.name = "value", variable.name = "condition") %>%
  as.data.frame()

wilcox.test(formula = value ~ condition, data = tsub %>% filter((marker == "Dysfunctional") & (condition %in% c("UT", des_cond))), 
            paired = T)
wilcox.test(formula = value ~ condition, data = tsub %>% filter((marker == "Predysfunctional") & (condition %in% c("UT", des_cond))), 
            paired = T)

colours = c("#AECAE4", "#CFD5EA", "#AF93BA", "#F1D687", "#D69E78", "#E68C7C", "#AECEB5", "#D6D6CE")
des_conditions = c("UT", des_cond)

p1 = ggplot(data = tsub %>% filter(condition %in% des_conditions), 
       mapping = aes(x = condition, y = value)) +
  geom_boxplot() +
  geom_jitter(height = 0, width = 0.25, mapping = aes(color = Patient_ID), size = 4) +
  labs(x = "Condition", y = "Percentage") +
  scale_color_manual(values = colours) +
  #+ coord_polar(start = 1)
  theme_bw() +
  theme(text = element_text(size = 20), legend.position = "none") +
  facet_wrap(~marker)
p1

#### 5C and 5E ####

## remove T cell subsets from radar 
rad = rad %>% filter(marker %ni% c("Dysfunctional", "Predysfunctional"))  
rad = rad[, c("Patient_ID", "UT", des_cond, "marker")]
colnames(rad)[colnames(rad) == des_cond] = "Treatment"

## Reshape data so each row = Patient ID and each column is a combo of Marker and UT/Treatment values
rad_wide = rad %>%  pivot_wider(names_from = marker, values_from = c(UT, Treatment))

## plot radar chart - fold changes or difference of treatment over UT ##
rad_wide = rad %>% group_by(marker) %>%
  summarise(med_treat = median(Treatment, na.rm = T),
            med_ut = median(UT, na.rm = T),
            mean_treat = mean(Treatment, na.rm = T),
            mean_ut = mean(UT, na.rm = T)) %>%
  mutate(med_treat_div_ut = med_treat / med_ut,
         mean_treat_div_ut = mean_treat / mean_ut) %>%
  select(c(marker, med_ut, med_treat)) %>%
  as.data.frame()

rownames(rad_wide) = rad_wide$marker
rad_wide$marker = NULL
rad_wide = rad_wide %>% t() %>% as.data.frame()

## Add max and minimum values for scaling
des_sheet = des_sheet[des_sheet %ni% grep("functional", des_sheet, value = T)]
max_min = as.data.frame(matrix(NA, ncol = length(des_sheet), nrow = 2))
colnames(max_min) = des_sheet

## pick max and min as: max() + 10% and min() - 10% of each marker
for (i in 1:length(des_sheet)) {
  curr_col = des_sheet[i]
  
  ## max
  max_val = max(rad_wide[, curr_col], na.rm = T)
  max_min[1, curr_col] = max_val + (0.1*max_val)
  
  ## min
  min_val = min(rad_wide[, curr_col], na.rm = T)
  max_min[2, curr_col] = min_val - (0.1*min_val)
}

## combine with the reshaped PDE data
plot_df = rbind(max_min, rad_wide)
colours = c("#685968", "#E96279")  

#pdf(paste0("RadarPlot_", des_cond, "_medianAcrossPatients_pctAndVal_functionalMarkers.pdf"), width = 10, height = 10)

fmsb::radarchart(plot_df, 
                 axistype = 3,                                                   # modify axis labels
                 vlcex = 1,                                                      # label size
                 vlabels = colnames(plot_df),                                    # labels
                 pcol = colours,                                                 # Colours for the lines
                 pfcol = scales::alpha(colours, 0.2),                            # colours for the filled areas
                 plwd = 2,                                                       # polygon line width
                 plty = 1,                                                       # polygon line type
                 cglcol = "black",                                               # colour of the grid
                 cglty = 1,                                                      # line type of grid
                 cglwd = 1,                                                      # line width of grid
                 axislabcol = "black",                                           # colour of axis labels
                 )

# add a legend
legend("topright", legend = c("Control", "Treated"), col = colours, lty = 1, lwd = 2)
dev.off()
