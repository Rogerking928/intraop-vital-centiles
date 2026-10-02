# GAMLSS sensitivity analysis for the reference centiles (as in de Graaff 2016 / WHO growth standards):
# for each variable and sex, fit BCCG, BCPE and BCT with P-splines of age in mu, sigma, nu (tau constant),
# keep the family with the lowest BIC, and predict the 3rd, 50th and 97th centiles at 20, 30, ..., 90 years.
# Input out/ref_for_gamlss.csv (written by gamlss_input.py); output out/gamlss_centiles.csv and out/gamlss_fit.csv,
# under PROJECT_DIR (default: the current directory).
suppressMessages(library(gamlss))
P <- Sys.getenv("PROJECT_DIR", ".")
d0 <- read.csv(file.path(P, "out", "ref_for_gamlss.csv"))
vars <- c("nibp_map", "nibp_sbp", "nibp_dbp", "hr", "etco2", "temp_c")
ages <- seq(20, 90, 10)
out <- list(); sel <- list()
for (v in vars) for (s in c("Female", "Male")) {
  d <- d0[d0$sex == s & !is.na(d0[[v]]), c("age", v)]
  names(d) <- c("age", "y")
  best <- NULL; bic <- Inf; fam_best <- NA
  for (fam in c("BCCG", "BCPE", "BCT")) {
    m <- tryCatch(gamlss(y ~ pb(age), sigma.formula = ~pb(age), nu.formula = ~pb(age),
                         family = fam, data = d, trace = FALSE, control = gamlss.control(n.cyc = 100)),
                  error = function(e) NULL)
    if (!is.null(m) && m$converged && GAIC(m, k = log(nrow(d))) < bic) {
      best <- m; bic <- GAIC(m, k = log(nrow(d))); fam_best <- fam }
  }
  cp <- centiles.pred(best, xname = "age", xvalues = ages, cent = c(3, 50, 97), plot = FALSE)
  out[[paste(v, s)]] <- data.frame(var = v, sex = s, age = ages, P3 = cp[, 2], P50 = cp[, 3], P97 = cp[, 4], family = fam_best)
  # in-sample share below each fitted centile
  z <- centiles.pred(best, xname = "age", xvalues = d$age, cent = c(3, 50, 97), plot = FALSE)
  sel[[paste(v, s)]] <- data.frame(var = v, sex = s, family = fam_best, n = nrow(d),
                                   below_P3 = mean(d$y < z[, 2]), below_P50 = mean(d$y < z[, 3]), below_P97 = mean(d$y < z[, 4]))
  cat(v, s, fam_best, "\n")
}
write.csv(do.call(rbind, out), file.path(P, "out", "gamlss_centiles.csv"), row.names = FALSE)
write.csv(do.call(rbind, sel), file.path(P, "out", "gamlss_fit.csv"), row.names = FALSE)
