import React from "react";
import { Textarea, Flex, IconButton } from '@chakra-ui/react'
import { AttachmentIcon, ArrowUpIcon } from '@chakra-ui/icons'
import ResizeTextarea from "react-textarea-autosize";
// import { INTERVIEW_STEP } from "@/utils/constants";
// import { useDispatch, useSelector } from "react-redux";
// import { setOpenUploadModal } from "@/store/slices/chatSlice";
// import { RootState } from "@/store";

interface InputBoxProps {
    message: string;
    setMessage: React.Dispatch<React.SetStateAction<string>>;
    sendMessage: () => void
}

const InputBox: React.FC<InputBoxProps> = ({ message, setMessage, sendMessage }) => {
    // const dispatch = useDispatch();
    // const { isTyping } = useSelector((state: RootState) => state.chat);
    const handleMessageChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
        setMessage(event.target.value);
    }

    return (
        <Flex
            zIndex="100"
            align="center"
            width='100%'
            padding="8px 15px"
            backgroundColor="#F0ECEB"
            borderRadius="50px"
        >
            <AttachmentIcon
                cursor="pointer"
                fontSize="18"
                // onClick={() => dispatch(setOpenUploadModal(true))}
            />
            <Textarea
                placeholder='Please answer the question'
                backgroundColor="transparent"
                resize="none"
                border="none"
                value={message}
                minRows={1}
                maxRows={3}
                as={ResizeTextarea}
                minH="unset"
                _focus={{ outline: 'none', boxShadow: 'none' }}
                // isDisabled={localStorage.getItem(INTERVIEW_STEP) === null || localStorage.getItem(INTERVIEW_STEP) === 'END'}
                onChange={handleMessageChange}
            />
            <IconButton aria-label='Send message' icon={<ArrowUpIcon />}  borderRadius="50px" ml={2} onClick={sendMessage} />
        </Flex>
    )
};

export default InputBox;