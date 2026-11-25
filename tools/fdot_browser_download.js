/*
 * FDOT crash downloader: Hillsborough County, 2014-2018.
 *
 * Why this exists: gis.fdot.gov sits behind a bot check that blocks scripts
 * (Python, curl), so the raw crash data has to be fetched from a real browser.
 * This script runs inside a browser tab that has already loaded a gis.fdot.gov
 * page, and saves one JSON file per year plus the layer metadata.
 *
 * How to run:
 *   1. Open https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer/0
 *      in Chrome (or Edge/Firefox) and wait until the layer page shows.
 *   2. Open the developer console (Chrome on Mac: Cmd+Option+J).
 *      If Chrome asks, type "allow pasting" and press Enter.
 *   3. Paste this whole file and press Enter. Progress prints in the console.
 *   4. Allow multiple downloads if the browser asks.
 *   5. Move the downloaded files from Downloads into data/raw/ in the repo.
 *
 * If it stops with "Got an HTML page", the bot check expired: reload the tab,
 * remove the finished years from YEARS below, and paste the script again.
 *
 * Output files:
 *   fdot_crashes_layer_metadata.json         field list, types, coded values
 *   fdot_crashes_hillsborough_<year>.json    {fields: [...], rows: [[...], ...]}
 */
(async () => {
  const LAYER = 'https://gis.fdot.gov/arcgis/rest/services/Crashes_All/FeatureServer/0';
  const YEARS = [2014, 2015, 2016, 2017, 2018];
  const CHUNK = 400;      // object IDs per request (keeps URLs short)
  const PAUSE_MS = 300;   // pause between requests, to be gentle on the server

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const t0 = Date.now();
  const elapsed = () => `${((Date.now() - t0) / 60000).toFixed(1)} min`;

  /** GET a JSON response from the ArcGIS REST API, with retries. */
  async function getJSON(url, params, tries = 5) {
    const qs = new URLSearchParams({ ...params, f: 'json' }).toString();
    for (let i = 1; i <= tries; i++) {
      try {
        const res = await fetch(`${url}?${qs}`, { credentials: 'include' });
        const text = await res.text();
        if (text.trimStart().startsWith('<')) {
          throw new Error('Got an HTML page instead of data (the bot check probably expired). ' +
            'Reload this tab, remove finished years from YEARS, and run the script again.');
        }
        const data = JSON.parse(text);
        if (data.error) throw new Error(`Server error: ${JSON.stringify(data.error)}`);
        return data;
      } catch (e) {
        if (i === tries || e.message.startsWith('Got an HTML')) throw e;
        console.warn(`Request failed (try ${i}/${tries}): ${e.message}. Retrying...`);
        await sleep(2000 * i);
      }
    }
  }

  /** Save an object as a JSON file through the browser's download mechanism. */
  function save(name, obj) {
    const blob = new Blob([JSON.stringify(obj)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    console.log(`Saved ${name} (${(blob.size / 1e6).toFixed(1)} MB)`);
  }

  // 1. Layer metadata: field names, types, coded-value domains, time reference.
  const meta = await getJSON(LAYER, {});
  const fields = meta.fields.map((f) => f.name);
  for (const need of ['COUNTY_TXT', 'CALENDAR_YEAR', 'INJSEVER']) {
    if (!fields.includes(need)) throw new Error(`Field ${need} not found. Fields: ${fields.join(', ')}`);
  }
  console.log(`Layer has ${fields.length} fields, maxRecordCount ${meta.maxRecordCount}`);
  console.log('Date time reference:', meta.dateFieldsTimeReference || '(not given)');
  save('fdot_crashes_layer_metadata.json', meta);

  // 2. Exact spelling of the county value.
  const distinct = await getJSON(`${LAYER}/query`, {
    where: "COUNTY_TXT LIKE '%ILLSB%'", outFields: 'COUNTY_TXT',
    returnDistinctValues: 'true', returnGeometry: 'false',
  });
  const counties = distinct.features.map((f) => f.attributes.COUNTY_TXT);
  console.log('County values found:', counties);
  if (counties.length !== 1) throw new Error('Expected exactly one Hillsborough county value; check the list above.');
  const county = counties[0];

  const yearType = meta.fields.find((f) => f.name === 'CALENDAR_YEAR').type;
  const yearValue = (y) => (yearType === 'esriFieldTypeString' ? `'${y}'` : y);

  // 3. One file per year: get every object ID, then fetch records in chunks.
  for (const year of YEARS) {
    const where = `COUNTY_TXT = '${county}' AND CALENDAR_YEAR = ${yearValue(year)}`;
    const idResp = await getJSON(`${LAYER}/query`, { where, returnIdsOnly: 'true' });
    const ids = (idResp.objectIds || []).sort((a, b) => a - b);
    console.log(`${year}: ${ids.length} crashes. Downloading... (${elapsed()})`);

    const rows = [];
    for (let i = 0; i < ids.length; i += CHUNK) {
      const part = await getJSON(`${LAYER}/query`, {
        objectIds: ids.slice(i, i + CHUNK).join(','),
        outFields: '*', returnGeometry: 'true', outSR: '4326',
      });
      for (const f of part.features) {
        rows.push([...fields.map((n) => f.attributes[n] ?? null),
          f.geometry ? f.geometry.x : null, f.geometry ? f.geometry.y : null]);
      }
      const done = Math.min(i + CHUNK, ids.length);
      if ((i / CHUNK) % 10 === 0 || done === ids.length) {
        console.log(`  ${year}: ${done} / ${ids.length} (${elapsed()})`);
      }
      await sleep(PAUSE_MS);
    }
    if (rows.length !== ids.length) {
      console.warn(`${year}: expected ${ids.length} rows but got ${rows.length}`);
    }
    save(`fdot_crashes_hillsborough_${year}.json`, {
      source: LAYER, where, downloaded_at: new Date().toISOString(),
      fields: [...fields, 'lon_wgs84', 'lat_wgs84'], rows,
    });
  }
  console.log(`All done in ${elapsed()}. Move the files into data/raw/.`);
})();
