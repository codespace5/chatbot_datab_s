import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ChatPage from "../pages/Chat";
import AppLayout from "../layout";
import DatabasePage from "../pages/Database";


const AppRouter = () =>{
    return(
        <Router>
            <Routes>
                <Route path="/" element = { <AppLayout><ChatPage /></AppLayout>  }/>
                <Route path="/database" element = { <AppLayout><DatabasePage /></AppLayout>  }/>
            </Routes>
        </Router>
    )
}

export default AppRouter