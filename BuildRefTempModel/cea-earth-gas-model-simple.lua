-- Earth representative air mixture for testing F&R stagnation heat flux code
-- Matt Uren (3/03/2023)

model = "CEAGas"

CEAGas = {
  mixtureName = 'earth',
  speciesList = {"O2","N2","O", "N", "NO"},
  reactants = {O2 = 0.21, N2=0.79},
  inputUnits = "moles",
  withIons = false,
  trace = 1.0e-20
}
