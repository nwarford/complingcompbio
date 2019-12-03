# using the structural topic model to infer mutation signatures
library(lda)
library(topicmodels)
library(stm)



######################
# helper functions   #
######################

# n : number of regions
generate_names <- function(n, s) {
  x <- vector(mode="character", length=n)
  for (i in 1:n) {
    x[i] <- paste0(s,as.character(i))
  }
  return(x)
}

generate_doc_names <- function(n) {
	return(generate_names(n, 'd'))
}

generate_word_names <- function(n) {
	return(generate_names(n, 'w'))
}

generate_region_names <- function(n) {
  return( generate_names(n, 'r') )
}

generate_cell_names <- function(n) {
  return( generate_names(n, 'c') )
}



##############################
# experimental protocols     #
##############################

name_rlda_gibbs_table <- function(k_vec, n_iters, tag="pbmc") {
  s = ""
  for (i in 1:length(k_vec)) {
    s = paste0(s, "_", k_vec[i])
  }
  s = paste0("rlda_gibbs_results_table", s, "_iter", n_iters, "_", tag, ".txt" )
  return(s)
}

# r lda gibbs sampler
# data as [region x cell] dgCMatrix
# k_vec, a vector of ks for model construction
rlda_gibbs_exp <- function(data, k_vec, n_iters=500) {

  # this code turns the data matrix into a formatted list of documents
  # necessary for rlda
  # Take binary count matrix
  object.binary.count.matrix <- data
  cellnames <- colnames(object.binary.count.matrix)
  regionnames <- rownames(object.binary.count.matrix)


  # Prepare data for r lda implementation
  print('Formatting data...')
  cellList <- split(as.integer(object.binary.count.matrix@i), rep(seq_along(object.binary.count.matrix@p+1), times=diff(c(object.binary.count.matrix@p+1, length(object.binary.count.matrix@i) + 1))))
  rm(object.binary.count.matrix)
  cellList <- lapply(cellList, function(x) {x <- rbind(x, rep(as.integer(1), length(x)))})
  names(cellList) <- cellnames
  cellList <- lapply(cellList, function(x) {colnames(x) <- regionnames[x[1,]+1];x})
  regionList <- regionnames

  ll_start = vector(mode="numeric", length=length(k_vec))
  ll_final = vector(mode="numeric", length=length(k_vec))

  # run LDA
  for (i in 1:length(k_vec)) {
    num_topics = k_vec[i]
    print(paste0("Running LDA w Gibbs sampler on ", num_topics, " topics."))
    model <- lda.collapsed.gibbs.sampler(cellList, num_topics, regionList, num.iterations=n_iters, alpha=50/num_topics, eta=0.1, compute.log.likelihood = TRUE, burnin=250)
    ll <- model$log.likelihoods
    ll_start[i] <- ll[2,1]
    ll_final[i] <- ll[2, n_iters]
  }
  results = data.frame(k_vec, ll_start, ll_final)
  write.table(results, name_rlda_gibbs_table(k_vec, n_iters))
}




name_rtopicmodels_vem_table <- function(k_vec, tag="pbmc") {
  s = ""
  for (i in 1:length(k_vec)) {
    s = paste0(s, "_", k_vec[i])
  }
  s = paste0("rtopicmodels_vem_results_table", s, "_", tag, ".txt" )
  return(s)
}

# variational gibbs sampler from topicmodels
# data as [Region X Cell] dgCMatrix
# k_vec, a vector of ks for model construction
rtopicmodels_vem_exp <- function(data, k_vec) {
  # convert to [Cells x Regions] (this implementation takes [doc x terms] matrix)
  print("transposing matrix")
  doc_term_mtx = Matrix::t(data)

  ll_final = vector(mode="numeric", length=length(k_vec))
  perp = vector(mode="numeric", length=length(k_vec))

  # run LDA
  for (i in 1:length(k_vec)) {
    num_topics = k_vec[i]
    print(paste0("Running VEM LDA on ", num_topics, " topics"))
    vem_model <- LDA(doc_term_mtx, k=num_topics, method="VEM")
    ll_final[i] = logLik(vem_model)[1]
    perp[i] = perplexity(vem_model)
  }
  results = data.frame(k_vec, ll_final, perp)
  write.table(results, name_rtopicmodels_vem_table(k_vec))
}


name_stm_vanilla_table <- function(k_vec, my_init, tag="pbmc") {
  s = ""
  for (i in 1:length(k_vec)) {
    s = paste0(s, "_", k_vec[i])
  }
  s = paste0("stm_vanilla_results_table_", my_init, "_init", s, "_", tag, ".txt" )
  return(s)
}

# run the LDA model from Structural Topic Model package without metadata
stm_transpose_vanilla_exp <- function(data, k_vec, my_init="Spectral") {
  # convert to [Cells x Regions] (this implementation takes [doc x terms] matrix)
  print("transposing matrix")
  doc_term_mtx = Matrix::t(data)

  ll_start = vector(mode="numeric", length=length(k_vec))
  ll_final = vector(mode="numeric", length=length(k_vec))

  # run LDA
  for (i in 1:length(k_vec)) {
    num_topics = k_vec[i]
    print(paste0("Running STM LDA on ", num_topics, " topics"))
    stm_model <- stm(documents=doc_term_mtx, K=num_topics, init.type=my_init)
    conv <- stm_model$convergence # readout of model likelihood
    
    ll_start[i] = conv$bound[1]
    ll_final[i] = conv$bound[length(conv$bound)]
  }
  results = data.frame(k_vec, ll_start, ll_final)
  write.table(results, "test_name_custom.txt")
  print("wrote table with custom name, trying template name")
  print(my_init)
  write.table(results, name_stm_vanilla_table(k_vec, my_init) )
}

# run the LDA model from Structural Topic Model package without metadata
stm_vanilla_exp <- function(doc_term_mtx, k_vec, my_init="Spectral") {

  ll_start = vector(mode="numeric", length=length(k_vec))
  ll_final = vector(mode="numeric", length=length(k_vec))

  # run LDA
  for (i in 1:length(k_vec)) {
    num_topics = k_vec[i]
    print(paste0("Running STM LDA on ", num_topics, " topics"))
    stm_model <- stm(documents=doc_term_mtx, K=num_topics, init.type=my_init)
    conv <- stm_model$convergence # readout of model likelihood
    
    ll_start[i] = conv$bound[1]
    ll_final[i] = conv$bound[length(conv$bound)]
  }
  results = data.frame(k_vec, ll_start, ll_final)
  write.table(results, "test_name_custom.txt")
  print("wrote table with custom name, trying template name")
  print(my_init)
  write.table(results, name_stm_vanilla_table(k_vec, my_init) )
}


# run the LDA model from Structural Topic Model package without metadata
stm_prevalence_exp <- function(doc_term_mtx, metadata, k_vec, formula, my_init="Spectral") {

  ll_start = vector(mode="numeric", length=length(k_vec))
  ll_final = vector(mode="numeric", length=length(k_vec))

  # run LDA
  for (i in 1:length(k_vec)) {
    num_topics = k_vec[i]
    print(paste0("Running STM LDA on ", num_topics, " topics"))
    stm_model <- stm(documents=doc_term_mtx, data=metadata, prevalence=formula, K=num_topics, init.type=my_init)
    conv <- stm_model$convergence # readout of model likelihood
    
    ll_start[i] = conv$bound[1]
    ll_final[i] = conv$bound[length(conv$bound)]
  }
  results = data.frame(k_vec, ll_start, ll_final)
  write.table(results, "test_name_custom.txt")
  print("wrote table with custom name, trying template name")
  print(my_init)
  write.table(results, name_stm_vanilla_table(k_vec, my_init) )
}

# run sbs vanilla baseline

vanilla_baseline_experiment <- function() {
	doc_term_mtx = load_sbs_data()
	vanilla_model = stm(doc_term_mtx,K=2, init.type="Random")
}



# run toy example

toy_example_experiment <- function(){
	doc_term_mtx = gen_toy_mtx()
	metadata = gen_toy_metadata()
	print("Toy Data:")
	print(doc_term_mtx)
	print("***")
	print("***")
	print("***")
	print("Toy Metadata:")
	print(head(metadata))
	print("...")
	print(tail(metadata))

	vanilla_model = stm(documents=doc_term_mtx, K=2, data=metadata, init.type="Random", verbose=F)
	print("***")
	print("***")
	print("***")
	print("Vanilla model topic contributions:")
	print(vanilla_model$theta)
	covariate_model = stm(documents=doc_term_mtx, K=2, data=metadata, prevalence=~xs, init.type="Random", verbose=F)

	print("Covariate model topic contributions:")
	print(covariate_model$theta)
}


#######################
# generate toy data   #
#######################

gen_toy_mtx <- function(){
	toy_data = matrix(nrow=22, ncol=6)
	rownames(toy_data) = generate_doc_names(22)
	colnames(toy_data) = generate_word_names(6)	

	for (i in 1:10) {
		toy_data[i,] = c(10,10,10,1,1,1)
	}
	for (i in 11:20) {
		toy_data[i,] = c(1,1,1,10,10,10)	
	}
	for (i in 21:22){
		toy_data[i,] = c(1,1,1,1,1,1)
	}
	return(as(toy_data, "dgCMatrix"))
}

gen_toy_metadata <- function(){
	xs = numeric(22)
	ys = numeric(22)
	cat = character(22)
	
	xs = c(sample(50:65, 10, replace=T), sample(1:5, 10, replace=T), sample(50:65, 2, replace=T))
	ys = c(sample(1:5, 10, replace=T), sample(50:65, 10, replace=T), sample(1:5, 2, replace=T))
	cat = c(rep("a", 10), rep("b", 10), rep("a", 2))

	df = data.frame(xs, ys, cat)
	rownames(df) = generate_doc_names(22)
	return(df)
}




####################
# reading data     #
####################

load_sbs_data <- function(){
	unformatted_df = read.table(file = "sbs_counts.tsv", sep="\t", header=TRUE)
	rownames(unformatted_df) = unformatted_df$X
	df = unformatted_df[-1]

	m = data.matrix(df, rownames.force=T)
	return(as(m, "dgCMatrix"))	
}


load_trimmed_cd4_atac <- function() {
  cd4 = as(Matrix::readMM("../data/trimmed_CD4_atac.mtx"), "dgCMatrix")
  return(cd4)
}

load_cd8_atac <- function() {
  cd8 = as(Matrix::readMM("../data/CD8_atac.mtx"), "dgCMatrix")
  return(cd8)
}



load_pbmc5k_atac_regnames <- function() {
  
}

load_pbmc5k_atac <- function() {
  pbmc = as(Matrix::readMM("../data/pbmc_5k_bin.mtx"), "dgCMatrix")
  cellnames = scan(file="../data/colnames_cellnames_pbmc_5k.txt", what=character())
  regnames = scan(file="../data/rownames_regnames_pbmc_5k.txt", what=character())
  colnames(pbmc) = cellnames
  rownames(pbmc) = regnames
  return(pbmc)
}



print(gen_toy_mtx())
print(gen_toy_metadata())
vanilla_baseline_experiment()



# [Regions x Cells]
#print("reading data")
#cd4 = as(Matrix::readMM("../data/trimmed_CD4_atac.mtx"), "dgCMatrix")

#pbmc = load_pbmc5k_atac()

#print("done reading data")

#rownames(cd4) = generate_region_names(dim(cd4)[1])
#colnames(cd4) = generate_cell_names(dim(cd4)[2])


#######################
# main                #
#######################
#print("starting experimental protocol")
#rlda_gibbs_exp(pbmc, c(1,2,3,5,10))
#rtopicmodels_vem_exp(pbmc, c(2,3,5,10))
#stm_vanilla_exp(pbmc, c(2,3,5,10), my_init="Spectral")



