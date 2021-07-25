const kunosCars = $.readText(`D:/Games/Assetto Corsa/content/sfx/GUIDs.txt`).split('\n').map(x => /cars\/(\w+)/.test(x) ? RegExp.$1 : null).unique().filter(x => x);
const carsDir = `D:/Games/Assetto Corsa/content/cars`;
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

async function bakeCar(carID, forceRebake) {
  console.log(β.yellow(`Baking 🚗 ${carID}…`));

  const mainModel = await findMainModelFilename(carID);
  const vaoPatch = path.join(carsDir, carID, 'main_geometry.vao-patch');
  const destination = path.join(destinationDir, `${carID}.vao-patch`);

  if (!forceRebake && fs.existsSync(vaoPatch) && fs.statSync(vaoPatch).mtime > 1607515255000) {
    console.log(β.grey(`💤 Existing VAO patch is new enough`));
    fs.copyFileSync(vaoPatch, destination);
    return;
  } else {
    await bakery(mainModel);
  }

  if (!fs.existsSync(vaoPatch)) {
    throw new Error('VAO patch is missing');
  }

  fs.copyFileSync(vaoPatch, destination);
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

// await bakeCars(...kunosCars);
// await bakeCars(...fs.readdirSync(carsDir).filter(x => /^ks_/.test(x)));
await bakeCars('ks_ferrari_488_gt3');
