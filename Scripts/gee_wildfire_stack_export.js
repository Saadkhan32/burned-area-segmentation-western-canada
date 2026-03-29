var wildfireEvents = [
  { lon: , lat: , date: '2024-05-05' }
];

function maskUsingSCL(img) {
  var scl = img.select('SCL');
  var keep =
      scl.neq(3)
      .and(scl.neq(8))
      .and(scl.neq(9))
      .and(scl.neq(10))
      .and(scl.neq(11));
  return img.updateMask(keep);
}

function scaleReflectance(img) {
  var scaled = img.select(['B2','B3','B4','B8','B11','B12'])
                  .multiply(0.0001)
                  .rename(['Blue','Green','Red','NIR','SWIR1','SWIR2']);
  return img.addBands(scaled, null, true);
}

function addIndices(img) {
  var NIR = img.select('NIR');
  var Red = img.select('Red');
  var SWIR1 = img.select('SWIR1');
  var SWIR2 = img.select('SWIR2');

  var ndvi = NIR.subtract(Red).divide(NIR.add(Red)).rename('NDVI');
  var nbr = NIR.subtract(SWIR2).divide(NIR.add(SWIR2)).rename('NBR');
  var ndmi = NIR.subtract(SWIR1).divide(NIR.add(SWIR1)).rename('NDMI');

  return img.addBands([ndvi, nbr, ndmi]);
}

var visRGB = {bands: ['Red','Green','Blue'], min: 0.02, max: 0.3, gamma: 1.2};

wildfireEvents.forEach(function (event) {

  var pt = ee.Geometry.Point([event.lon, event.lat]);
  var buffer = pt.buffer(10000);
  var region = buffer.bounds();
  var startDate = ee.Date(event.date);

  var lookDays = 30, maxWindows = 12, cursor = startDate, bestImage = null;

  for (var i = 0; i < maxWindows; i++) {
    var endDate = cursor.advance(lookDays, 'day');

    var col = ee.ImageCollection('COPERNICUS/S2_SR')
      .filterBounds(buffer)
      .filterDate(cursor, endDate)
      .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 5))
      .map(maskUsingSCL)
      .map(scaleReflectance)
      .map(addIndices);

    if (col.size().getInfo() > 0) {
      bestImage = ee.Image(col.sort('CLOUDY_PIXEL_PERCENTAGE').first());
      print('Found low-cloud S2 scene between',
            cursor.format('YYYY-MM-dd').getInfo(), 'and',
            endDate.format('YYYY-MM-dd').getInfo());
      break;
    }
    cursor = endDate;
  }

  if (!bestImage) {
    print('No ≤5% cloud scenes — using 30-day median composite for', event.date);
    bestImage = ee.ImageCollection('COPERNICUS/S2_SR')
      .filterBounds(buffer)
      .filterDate(startDate, cursor)
      .map(maskUsingSCL)
      .map(scaleReflectance)
      .map(addIndices)
      .median();
  }

  var stack = bestImage.select([
    'Blue','Green','Red','NIR','SWIR1','SWIR2','NDVI','NBR','NDMI'
  ]).clip(region)
   .resample('bilinear');

  Map.centerObject(buffer, 10);
  Map.addLayer(stack, visRGB, 'True-Color (scaled) ' + event.date);
  Map.addLayer(stack.select('NDVI'), {min:-0.2, max:0.8}, 'NDVI ' + event.date, false);
  Map.addLayer(stack.select('NBR'), {min:-0.5, max:0.5}, 'NBR ' + event.date, false);

  Export.image.toDrive({
    image: stack.toFloat(),
    description: 'S2_WildfireStack_' + event.date.replace(/-/g, ''),
    folder: 'GEE_Exports/WildfireStack',
    fileNamePrefix: 'S2_WildfireStack_' + event.date.replace(/-/g, ''),
    region: region,
    scale: 10,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });

});