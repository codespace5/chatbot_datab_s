
// import { Provider } from "react-redux"
import { ChakraProvider } from "@chakra-ui/react"
import AppRouter from "./routes"
// import { store } from "./store"

function App() {

  return (
    // <Provider store={store}>
      <ChakraProvider>
        <AppRouter/>
      </ChakraProvider>
    // </Provider>
  )
}

export default App
