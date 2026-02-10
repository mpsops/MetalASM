/// A basic block in an LLVM IR function.
public final class IRBasicBlock {
    /// The label name of this basic block.
    public var name: String

    /// The instructions in this basic block.
    public var instructions: [IRInstruction]

    public init(name: String, instructions: [IRInstruction] = []) {
        self.name = name
        self.instructions = instructions
    }
}
