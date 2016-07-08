
iris.plt <- cbind.data.frame(sapply(iris[,1:4], rescale), 
                             Species = iris$Species,
                             ID = seq_len(nrow(iris)))
iris.plt <- melt(iris.plt, id.vars = c("Species", "ID"))
pcolors <- brewer.pal(length(levels(iris$Species)), "Set1")
pcolors <- adjustcolor(pcolors, alpha = 0.7)
ggplot(iris.plt) +
  geom_line(aes(x = variable, y = value, color = Species, group = ID),
            size = 0.65) +
  scale_color_manual(values = pcolors) +
  theme_bw()
