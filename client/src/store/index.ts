import { combineReducers, configureStore } from "@reduxjs/toolkit";

// import authSlice from "./slices/authSlice";
import chatSlice from "./slices/chatSlice";

const rootReducer = combineReducers({
    chat: chatSlice
});

export const store = configureStore({
    reducer: rootReducer
});

export type RootState = ReturnType<typeof store.getState>;

export type AppDispatch = typeof store.dispatch;