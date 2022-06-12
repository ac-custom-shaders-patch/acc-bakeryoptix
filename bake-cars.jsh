const $zipMerger = $['C:/Apps/Utils/ZipMerger.exe'];
// const $zipMerger = $['D:/Documents/Visual Studio 2015/Projects/Utils/ZipMerger/bin/Release/ZipMerger.exe'];
const acRoot = `D:/Temporary/Games/AssettoCorsa`;
const kunosCars = $.readText(`${acRoot}/content/sfx/GUIDs.txt`).split('\n')
  .map(x => /cars\/(\w+)/.test(x) ? RegExp.$1 : null).unique().filter(x => x);
const carsDir = `${acRoot}/content/cars`;
const extraCars = [];
const destinationDir = `C:/Development/acc-extension-cars-vao`;
const bakery = $[`${__dirname}/x64/Release/bakeryoptix.exe`];

async function findMainModelFilename(carID) {
  const dir = path.join(carsDir, carID);
  if (!fs.existsSync(dir)) {
    throw new Error('Car is missing: ' + dir);
  }

  const kn5s = fs.readdirSync(dir).filter(x => /\bkn5\b/i.test(x)).map(x => ({
    name: x,
    size: fs.statSync(path.join(dir, x)).size
  })).filter(x => x.size > 0);

  if (kn5s.length === 0) {
    throw new Error('No models here');
  }

  if (kn5s.length !== 5) {
    $.cd(dir);
    await $.explorer('.', { fail: false });
    throw new Error('Unexpected amount of models: ' + kn5s.map(x => x.name).join(', '));
  }

  const largest = kn5s.reduce((a, b) => a.size > b.size ? a : b, kn5s[0]);
  return path.join(dir, largest.name);
}

function applyVaoPatch(vaoPatch, destination){
  if (fs.existsSync(destination)){
    $zipMerger(destination, destination, vaoPatch);
  } else {
    fs.copyFileSync(vaoPatch, destination);
  }
}

async function bakeCar(carID, forceRebake) {
  console.log(β.yellow(`Baking 🚗 ${carID}…`));

  const mainModel = await findMainModelFilename(carID);
  const vaoPatch = path.join(carsDir, carID, 'main_geometry.vao-patch');
  const destination = path.join(destinationDir, `${carID}.vao-patch`);

  if (!forceRebake && fs.existsSync(vaoPatch) && fs.statSync(vaoPatch).mtime > 1628685545787) {
    console.log(β.grey(`💤 Existing VAO patch is new enough`));
    applyVaoPatch(vaoPatch, destination);
    return;
  } else {
    await bakery(mainModel);
  }

  if (!fs.existsSync(vaoPatch)) {
    throw new Error('VAO patch is missing');
  }

  applyVaoPatch(vaoPatch, destination);
  console.log(β.green(`✅ Done, VAO patch “${path.basename(destination)}” is ready`));
}

async function bakeCars(...carIDs) {
  for (const carID of carIDs) {
    try {
      await bakeCar(carID, carIDs.length === 1);
    } catch (e) {
      $.echo(β.red(`❌ ${e}`));
    }
  }
}

$.echo('Script mtime: ' + +fs.statSync(`${__dirname}/bake-cars.jsh`).mtime);

await bakeCars(...kunosCars);
// await bakeCars(...fs.readdirSync(carsDir).filter(x => /^ks_/.test(x)));
// await bakeCars('ks_ferrari_488_gt3');
// await bakeCars('ks_porsche_962c_longtail', 'ks_porsche_962c_shorttail');
// await bakeCars('ks_porsche_962c_longtail');
