gmt begin Fig jpg

REM #######################Detrend#####################
	REM gmt grdtrend Bathymetry.grd -N6 -TBath_trend.grd -DBath_detrend.grd 
	REM gmt grdtrend GA.grd -N6 -TGA_trend.grd -DGA_detrend.grd 
	REM gmt grdtrend VGG.grd -N6 -TVGG_trend.grd -DVGG_detrend.grd 
	
REM ######################Cohenrency####################	
	REM gmt grdfft Bath_detrend.grd GA_detrend.grd -Er+n -N+l -fg > cross_spectra_ga.txt
	REM gmt grdfft Bath_detrend.grd VGG_detrend.grd -Er+n -N+l -fg > cross_spectra_vgg.txt	

REM ###############Data processing for linear regression#######
	REM gmt grdfft Bath_detrend.grd -F-/40000 -fg -GTemp1.grd
	REM gmt grdfft GA_detrend.grd -F-/40000 -fg -GTemp2.grd
	REM gmt grdfft VGG_detrend.grd -F-/40000 -fg -GTemp3.grd

	REM gmt grdtrack SIOv20.1.txt -GTemp1.grd -GTemp2.grd -GTemp3.grd -GGA.grd -GVGG.grd> ss.txt
	
REM ###################Prediction fature data################
	REM gmt surface training_dataset.txt -i0,1,5 -R140/148/10/16 -I10s -GTemp4.grd
	REM gmt surface training_dataset.txt -i0,1,6 -R140/148/10/16 -I10s -GTemp5.grd
	
	REM gmt grdtrack SIOv20.1.txt -GGA.grd -GVGG.grd -GTemp4.grd -GTemp5.grd> Prediction_dataset.txt
gmt end show